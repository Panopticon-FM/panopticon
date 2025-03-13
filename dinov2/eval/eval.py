import os
from dinov2.eval.wrapper import backbone_to_features
from dinov2.data.loaders import make_dataset
from dinov2.eval.setup import parse_model_obj, setup_logger, _setup, merge_cfgs
from dinov2.eval.linear import run_eval_linear, plot_curves
from dinov2.eval.knn import eval_knn_with_model
from functools import partial
from omegaconf import OmegaConf
import pandas as pd
from dinov2.configs import default_eval_linear_config, default_eval_knn_config, default_eval_linear_multilabel_config
import fire
import time
from dinov2.eval.setup import parse_config_obj
import itertools
import dinov2.distributed as distributed
import torch.distributed as dist
import numpy as np
import wandb
import logging
from dinov2.logging import setup_logging as setup_train_logger
from copy import deepcopy
from omegaconf import OmegaConf


def do_linear(model_cfg=None, task_cfg=None, output_dir=None):
    """ actually need all args, but have kwargs for flexible functools.partial here"""
    model = _setup(model_cfg, task_cfg, output_dir)

    # execute training
    results_dict = run_eval_linear(
        model=model,
        output_dir=output_dir,

        train_dataset_cfg=task_cfg.train_dataset,
        val_dataset_cfg=task_cfg.val_dataset,
        test_dataset_cfgs=task_cfg.test_datasets_list,

        dl_cfg = task_cfg.optim.dl,

        epochs=task_cfg.optim.epochs,
        iter_per_epoch=task_cfg.optim.iter_per_epoch,
        save_checkpoint_frequency_epoch=task_cfg.optim.save_checkpoint_frequency_epoch,
        eval_period_epoch=task_cfg.optim.eval_period_epoch,
        eval_period_iter=task_cfg.optim.eval_period_iter,

        heads=task_cfg.heads,

        val_metrics=task_cfg.task.val_metrics,
        test_metrics_list=task_cfg.task.test_metrics_list,
        criterion_cfg = task_cfg.task.criterion_cfg,

        # add dinov2 eval args, not sure why
        resume=not task_cfg.no_resume,
        classifier_fpath=task_cfg.classifier_fpath,
        val_class_mapping_fpath=task_cfg.val_class_mapping_fpath,
        test_class_mapping_fpaths=task_cfg.test_class_mapping_fpaths,)

    # print results to csv
    results_dict = [dict(
        value = d['val'],
        metric = d['metric_str'],
        best_classifier = d['name']
    ) for d in results_dict]
    df = pd.DataFrame(results_dict).set_index(['metric'])
    with open(os.path.join(output_dir, 'results.csv'),'w') as f:
        df.to_csv(f)

    # plot training curves
    plot_curves(output_dir, suppress_print=True)


def do_knn(model_cfg=None, task_cfg=None, output_dir=None):
    """ actually need all args, but have kwargs for flexible functools.partial here"""
        # this corresponds to vanilla DINOv2 ModelWithNormalize

    # build model
    model = _setup(model_cfg, task_cfg, output_dir)
    bb_to_feat_adapter = partial(
        backbone_to_features, **task_cfg.task.backbone_to_features)
    model.set_n_last_blocks(task_cfg.task.backbone_to_features.use_n_blocks)
    model.bb_to_feat_adapter = bb_to_feat_adapter

    # config equals train_config.evaluation
    train_dataset = make_dataset(task_cfg.train_dataset, seed=task_cfg.seed)
    val_dataset = make_dataset(task_cfg.test_dataset, seed=task_cfg.seed) # although it is called val_dataset, this is actually the test_dataset

    # actual computations
    results_dict = eval_knn_with_model(
        model=model,
        output_dir=output_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        nb_knn=task_cfg.heads.nb_knn,
        temperature=task_cfg.heads.temperature,
        metric_cfg=task_cfg.task.metrics,
        gather_on_cpu=task_cfg.heads.gather_on_cpu,
        n_per_class_list=task_cfg.heads.n_per_class_list,
        n_tries=task_cfg.heads.n_tries,
        dl_cfg=task_cfg.optim.dl,
        is_multilabel=task_cfg.task.is_multilabel,)

    # print results to csv
    df = pd.DataFrame(results_dict).set_index(['metric'])
    with open(os.path.join(output_dir, 'results.csv'),'w') as f:
        df.to_csv(f)


def do_debug(**kwargs):
    os.makedirs(kwargs['output_dir'], exist_ok=True)
    out = []
    for metric_str in kwargs['task_cfg'].metrics:
        out.append(dict(
            value = np.random.rand(),
            metric = metric_str,
            best_classifier = f'myclassifier_{np.random.randint(100)}'
        ))
    df = pd.DataFrame(out).set_index(['metric'])
    with open(os.path.join(kwargs['output_dir'], 'results.csv'),'w') as f:
        df.to_csv(f)
    return out


def build_task(task_cfg, **overwrites):
    id = task_cfg.task.id
    if id == 'classification':
        default_cfg = default_eval_linear_config
        fct = do_linear

    elif id == 'multilabelclassification':
        default_cfg = default_eval_linear_multilabel_config
        fct = do_linear

    elif id == 'knn':
        default_cfg = default_eval_knn_config
        fct = do_knn

    elif id == 'debug':
        default_cfg = OmegaConf.create({})
        fct = do_debug

    else: 
        raise ValueError(f'Unknown task id {id}')

    task_cfg = merge_cfgs(default_cfg, task_cfg, dotpath_to_dict(overwrites))
    task = partial(fct, task_cfg=task_cfg)
    return task

def _get_wandb_id_from_dir(wandb_dir):
    files = os.listdir(os.path.join(wandb_dir,'wandb','latest-run'))
    files = [f for f in files if f.startswith('run-') and f.endswith('.wandb')]
    assert len(files) == 1
    return files[0][4:-6]

def main(model_obj, 
         config_obj = None, 
         output_dir = None, 
         overwrite = False, 
         use_wandb = True, 
         wandb_mode = 'api', 
         distribute = 'runs', 
         barrier = True,
         summary_logger = None,
         log_file = None,
         only_avg = False,
         **overwrites):
    """ handles all evaluation for single model, 
        config_obj should be one of: path to .yaml file, path to folder, or multiple of the objects before separated by ; """

    if not distributed.is_enabled(): # fct directly called from fire => do minimal setup
        distributed.enable(overwrite=True)
        summary_logger = setup_logger('eval', filename=log_file, to_sysout=True)

    # get models

    model_tuples = parse_model_obj(model_obj)
    if len(model_tuples) > 1:
        assert output_dir is None, "cannot save different models in same (provided) output_dir"
    elif output_dir is not None:
        model_tuples[0] = (output_dir, *model_tuples[0][1:])

    # build tasks

    task_tuples = parse_config_obj(config_obj)
    task_tuples = [(c[0], build_task(c[1], **overwrites)) for c in task_tuples]

    # run tasks

    if only_avg:
        assert config_obj is None, 'Cannot compute tasks in only_avg mode'
        all_tasks_done = distributed.is_main_process()
    else:
        all_tasks_done = _distribute_and_execute(
            model_tuples, 
            task_tuples, 
            overwrite = overwrite, 
            distribute = distribute, 
            config_obj_str = config_obj, 
            barrier = barrier,
            summary_logger = summary_logger)

    # upload results to wandb

    if all_tasks_done:

        average_to_metric = OmegaConf.to_container(OmegaConf.load(
            os.path.join(os.environ['CDIR'],'defaults','wandb_averages.yaml')))
        
        for output_dir, _, _, wandb_dir in model_tuples:
            if use_wandb and wandb_dir != '' and os.path.exists(os.path.join(wandb_dir,'wandb')):

                wandb_id = _get_wandb_id_from_dir(wandb_dir)

                # get wandb train run
                if wandb_mode == 'api': # doesn't affect runtime but seems buggy
                    # logger.info(f'Uploading to wandb with api (id={wandb_id}) ...')
                    run = wandb.Api().run(f'{wandb_id}')
                elif wandb_mode == 'resume': # bugs in resuming
                    finish_run = True
                    if wandb.run is not None:
                        assert wandb.run.id == wandb_id
                        finish_run = False
                    else:
                        wandb.init(resume='must', id=wandb_id, dir=wandb_dir)
                else:
                    raise ValueError(f'Unknown wandb_mode {wandb_mode}')

                # get relpath
                df = collect_csv(output_dir, return_df='relpath')
                df['log_key'] = [os.path.join(df.at[i,'relpath'],df.at[i,'metric']) 
                            for i in range(len(df))]
                df = df.drop(columns=['relpath','metric','best_classifier'])

                # add all average
                df = pd.concat([df, pd.DataFrame([{
                    'log_key': '_avg', 'value': round(df['value'].mean(),2)}])])
                df = df.reset_index(drop=True)

                # add specific averages
                average_out = {n: [] for n in average_to_metric.keys()}
                for name, metrics in average_to_metric.items():
                    for i in range(len(df)):
                        if df.at[i,'log_key'] in metrics:
                            average_out[name].append(df.at[i,'value'])
                            # print(f'{df.at[i,"log_key"]} -> {name}')
                    average_out[name] = np.mean(average_out[name])

                for name, value in average_out.items():
                    df = pd.concat([df, pd.DataFrame([{
                        'log_key': name, 'value': round(value,2)}])])
                
                # actually log
                df = df.reset_index(drop=True)
                for i in range(len(df)):
                    key = df.at[i,'log_key']
                    value = df.at[i,'value']
                    if only_avg and not key.startswith('_'):
                        continue

                    if wandb_mode == 'api':
                        run.summary[key] = value
                        run.update()
                    elif wandb_mode == 'resume':
                        wandb.run.summary[key] = value

                if wandb_mode == 'resume' and finish_run:
                    wandb.run.finish()
                # elif wandb_mode == 'api':
                    # logger.info(f'Finished uploading to wandb with api (id={wandb_id})')

    return all_tasks_done

def _check_if_all_tasks_done(model_tuples, task_tuples):
    all_tasks_done = True
    for base_output_dir, _, _, _ in model_tuples:
        df = collect_csv(base_output_dir, return_df='relpath')
        for relpath, _ in task_tuples:
            if not relpath in df['relpath'].values:
                all_tasks_done = False
                break
        if not all_tasks_done:
            break
    return all_tasks_done

def _wait_until_all_tasks_done(model_tuples, task_tuples, sleep=5):
    all_tasks_done = False
    while not all_tasks_done:
        time.sleep(sleep)
        all_tasks_done = _check_if_all_tasks_done(model_tuples, task_tuples)
    return True


def _distribute_and_execute(
        model_tuples, 
        task_tuples, 
        distribute = 'models', 
        overwrite = False, 
        config_obj_str = None, 
        barrier = True, 
        summary_logger : logging.Logger = None ):
    """ Executes cross product of model and task tuples by distributing each task 
        to a single GPU. Torch.distributed is blocked in eval by setting 
        distributed._ARTIFICIALLY_BLOCK_DISTRIBUTED.

        Notes:
        - barrier: if false returns single all_tasks_done check, if true all workers
            wait until all tasks are done and only main proc returns true
        - config_obj_str is used for logging only 
        - logging: process overview logged to summary_logger, progress of individual tasks
                    logged to file in output_dir"""
    assert summary_logger is not None, 'Need to provide summary_logger'

    # cache dinov2 logger handlers (since dinov2 re-directed during eval, need to attach them to some logger)

    dinov2_logger = logging.getLogger('dinov2')
    cached_dinov2_logger = logging.getLogger('cached_dinov2') 
    for h in cached_dinov2_logger.handlers:
        cached_dinov2_logger.removeHandler(h)
    for h in dinov2_logger.handlers:
        cached_dinov2_logger.addHandler(h)
        dinov2_logger.removeHandler(h)

    # setup summary logger as copy from dinov2 logger

    logger = summary_logger
    if distributed.is_main_process():
        logger.info(f'-------------- eval ---------------')
        logger.info(f'#models:    {len(model_tuples)}')
        logger.info(f'#configs:   {len(task_tuples)}')
        logger.info(f'#nworkers:  {distributed.get_global_size()}')
        logger.info(f'distribute: {distribute}')
        logger.info(f'overwrite:  {overwrite}')
        logger.info(f'barrier:    {barrier}')
        logger.info(f'config_obj: {config_obj_str}')
        logger.info('config_relpaths:')
        for i,t in enumerate(task_tuples):
            logger.info(f'  t{i:03d}: {t[0]}')
        logger.info('model_objs:') 
        for i,t in enumerate(model_tuples):
            logger.info(f'  m{i:03d}: {t[2]}')
        logger.info('------------------------------------')
    time.sleep(0.1)

    # setup logger for each model_tuple

    loggers = []
    summary_logger_name = summary_logger.name
    for i, (base_output_dir, model_cfg, model_str, _) in enumerate(model_tuples):

        os.makedirs(base_output_dir, exist_ok=True)
        logfile = os.path.join(base_output_dir, 'log')

        appending_log_file = False
        if os.path.exists(logfile):
            with open(logfile,'r') as f:
                appending_log_file = len(f.read(1)) > 0

        logger = setup_logger(f'{summary_logger_name}.m{i:03d}', logfile, to_sysout=False)
        logger.propagate = False

        if distributed.is_main_process():
            if appending_log_file:
                with open(logfile,'a') as f:
                    f.write('\n\n\n\n')

            logger.info(f'------------------------------------')
            logger.info(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, {os.environ["USER"]}')
            logger.info(f'model-obj:  {model_str}') # uniquely identifies model
            if config_obj_str is not None:
                logger.info(f'config-obj: {config_obj_str}')
            logger.info(f'output-dir: {base_output_dir}')
            logger.info(f'overwrite:  {overwrite}')
            logger.info(f'------------------------------------')

        logger.propagate = True
        loggers.append(logger)

    # assign tasks to rank

    if distribute == 'models':
        models_rank = itertools.islice(
            range(len(model_tuples)), 
            distributed.get_global_rank(), 
            len(model_tuples), 
            distributed.get_global_size())
        task_indices_rank = itertools.product(range(len(task_tuples)), models_rank)

    elif distribute == 'runs':
        ntasks = len(model_tuples) * len(task_tuples)
        tasks = itertools.product(range(len(task_tuples)), range(len(model_tuples)))
        task_indices_rank = itertools.islice(
            tasks, 
            distributed.get_global_rank(), 
            ntasks, 
            distributed.get_global_size())
        
    else:
        raise ValueError(f'Unknown distribute {distribute}')

    # pause torch distributed pg

    rank = distributed.get_global_rank()
    size = distributed.get_global_size()
    distributed._ARTIFICIALLY_BLOCK_DISTRIBUTED = True

    # work tasks

    for task_idx, model_idx in task_indices_rank:

        base_output_dir, model_cfg, _, _ = model_tuples[model_idx]
        task_name, task = task_tuples[task_idx]
        logger = logging.getLogger(f'{summary_logger_name}.m{model_idx:03d}')

        task_output_dir = os.path.join(base_output_dir, task_name)
        if os.path.exists(os.path.join(task_output_dir, 'results.csv')) and not overwrite:
            logger.info(f'Skipping {task_name} as results.csv already exists (rank {rank}/{size})')
            continue

        logger.info(f'Running  {task_name} ... (rank {rank}/{size})')
        start = time.time()
        task(model_cfg=deepcopy(model_cfg), output_dir=task_output_dir)
        logger.info(f'Finished {task_name} in {time.time()-start:.2f}s (rank {rank}/{size})')

        df = collect_csv(base_output_dir, return_df='all')
        model_done = True
        for name, _ in task_tuples:
            if not name in df['relpath'].values:
                model_done = False
                break
        if model_done:
            logger.propagate = False
            logger.info(f'All tasks done.')
            logger.info(f'\n{df.drop(columns=["relpath"]).to_string()}')
            summary_logger.info(f'All tasks done for model {model_idx:03d}.')

    # resume torch distributed pg

    distributed._ARTIFICIALLY_BLOCK_DISTRIBUTED = False

    # wait / check if all tasks done

    if barrier:
        _wait_until_all_tasks_done(model_tuples, task_tuples, sleep=5)
        all_tasks_done = distributed.is_main_process()
        dist.barrier()  # barrier to potentially sync the 5s diff from above
    else:
        all_tasks_done = _check_if_all_tasks_done(model_tuples, task_tuples)
    if all_tasks_done:
        summary_logger.info('All tasks done.')

    # reinstantiate original dinov2 logger

    dinov2_logger = logging.getLogger('dinov2')
    for h in cached_dinov2_logger.handlers:
        dinov2_logger.addHandler(h)
        cached_dinov2_logger.removeHandler(h)

    return all_tasks_done



def collect_csv(output_dir, return_df='relpath', sleep=0.05):
    """ recursively create all aggregation csv files in nested folder structure """
    
    def add_lvl(df: pd.DataFrame):
        """ for pretty to_string() convert relpath to index with lvl0, lvl1, ... """
        if df.shape[0] == 0:
            return df
        df['lvl'] = df['relpath'].apply(lambda x: x.split('/'))
        max_level = max([len(l) for l in df['lvl']])
        for i in range(max_level):
            df[f'lvl{i}'] = df['lvl'].apply(lambda x: x[i] if len(x) > i else '')
        df.set_index([f'lvl{i}' for i in range(max_level)] + ['metric'], inplace=True)
        df.drop(columns=['lvl'], inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def rec(abspath):
        sub_dirs = [p for p in os.listdir(abspath) if os.path.isdir(os.path.join(abspath,p))]
        sub_results = [rec(os.path.join(abspath,p)) for p in sub_dirs]
        sub_results = [r for r in sub_results if len(r) > 0]

        if len(sub_results) == 0:
            if 'results.csv' in os.listdir(abspath):
                if sleep > 0.0:
                    time.sleep(sleep)
                df = pd.read_csv(os.path.join(abspath, 'results.csv'))
                df['relpath'] = ''
                return (os.path.basename(abspath), df)
            return ()
        
        else:
            dfs = []
            for d, df in sub_results:
                df['relpath'] = df['relpath'].apply(lambda x: os.path.normpath(os.path.join(d,x)))
                dfs.append(df)
            df = pd.concat(dfs).reset_index(drop=True)

            df.to_csv(os.path.join(abspath, 'results.csv'))
            with open(os.path.join(abspath, 'results.txt'),'w+') as f:
                df_print = add_lvl(df.copy()).drop(columns=['relpath'])
                f.write(df_print.to_string())

            return (os.path.basename(abspath), df)
                
    t = rec(output_dir)
    if len(t) == 0:
        df = pd.DataFrame(columns=['metric','value','best_classifier','relpath'])
    else:
        df = t[1]
    
    if return_df == 'lvl':
        return add_lvl(df).drop(columns=['relpath'])
    elif return_df == 'relpath':
        return df
    elif return_df == 'all':
        return add_lvl(df)
    else:
        raise ValueError(f'Unknown return_df {return_df}')


def dotpath_to_dict(dotpath):
    out = {}
    for k,v in dotpath.items():
        keys = k.split('.')
        d = out
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return out

if __name__ == '__main__':
    fire.Fire()