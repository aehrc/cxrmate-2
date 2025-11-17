import datetime
import glob
import math
import multiprocessing
import os
import re
import string
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotnine as p9
import seaborn
from pypdf import PdfReader, PdfWriter

SHAPES = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]


def save_plot_to_memory(plot):
    buffer = BytesIO()
    plot.save(buffer, format='pdf', verbose=False, limitsize=False)
    buffer.seek(0)
    return buffer


def concatenate_and_save_pdfs(plot_buffers, output_path):
    pdf_writer = PdfWriter()
    for buffer in plot_buffers:
        pdf_reader = PdfReader(buffer)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])
    with open(output_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)


def read_and_format_csv(csv_path, trial, config, datetime, columns=None):
    if columns is not None:
        csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        columns = [column for column in csv_columns if column in columns]
        df = pd.read_csv(csv_path, usecols=columns)
    else:
        df = pd.read_csv(csv_path)
    columns = df.columns.to_list()
    if 'epoch' not in columns:
        return pd.DataFrame()
    if 'step' not in columns:
        return pd.DataFrame()
    # df = df[df['epoch'] != -1]
    columns.remove('epoch')
    columns.remove('step')
    df = df.dropna(how='all', subset=columns)
    df.insert(0, 'trial', trial) 
    df.insert(0, 'config', config) 
    df = df.melt(
        id_vars=['config', 'trial', 'stage', 'epoch', 'step', 'scheduler_step'] if 'scheduler_step' in df.columns else ['config', 'trial', 'stage', 'epoch', 'step'],
        var_name='metric',
        value_name='score',
    )
    df = df.dropna(subset=['score'])
    df['datetime'] = datetime
    return df


def get_config_scores(config_trial_list, columns):

    num_workers = multiprocessing.cpu_count()
    
    csv_paths = []
    for i in config_trial_list:
        csv_path = os.path.join(i['config'], f'trial_{i['trial']}', 'metrics.csv')
        file_datetime = datetime.datetime.fromtimestamp(os.path.getctime(csv_path))
        csv_paths.append({'csv_path': csv_path, 'datetime': file_datetime, 'trial': i['trial'], 'config': i['config']})
    csv_paths.sort(key=lambda x: x['datetime'])
    
    def read_and_format_csv_(entry):
        return read_and_format_csv(entry['csv_path'], entry['trial'], entry['config'], entry['datetime'], columns)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        df_list = list(executor.map(read_and_format_csv_, csv_paths))

    df_list = [i for i in df_list if not i.empty]
    df = pd.concat(df_list, ignore_index=True)

    df_filter = pd.DataFrame(config_trial_list)
    df = pd.merge(df, df_filter, on=['config', 'trial'])

    return df


def get_config_colours(unique_configs, unique_config_trials):
    colours = seaborn.color_palette(palette='hls', n_colors=len(unique_configs))
    colours = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colours]
    config_colours = {k:v for k, v in zip(unique_configs, colours)}
    config_colours = {**config_colours, **{i:config_colours[re.sub(r'\s*trial\s*\d+', '', i)] for i in unique_config_trials}}
    return config_colours


def get_config_scores_dataframe(configs, ignore_trials=None):
                
    plot_buffers = []
    
    config_list = [config['path'] for config in configs]
    trial_paths = [j for i in config_list for j in glob.glob(i + '/trial_*/metrics.csv', recursive=True)]
    config_trial_list = []
    for i in trial_paths:
        parent_dir = Path(i).parents[1]
        config_check = [
            cfg for cfg in config_list
            if Path(cfg).resolve() == parent_dir.resolve()
        ]
        assert len(config_check) <= 1, f'Multiple configs were in the following path: {i}.'
        if config_check:
            match = re.search(r'trial_(\d+)', i)
            if match:
                trial = int(match.group(1))
                config_trial_list.append({'config': config_check[0], 'trial': trial})
        
    # Get scores from .csv files:
    df = get_config_scores(config_trial_list=config_trial_list, columns=None)
    
    # Format and sort dataframe by configuration:
    df['config'] = df['config'].replace({i['path']: i['name'] for i in configs})
    df['config'] = pd.Categorical(df['config'], categories=[i['name'] for i in configs])
    df = df.sort_values(['config'])
    df = df[df['config'].notna()]
    df['config_trial'] = df['config'].astype(str) + ' trial ' + df['trial'].astype(str) 
    
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    return df
    