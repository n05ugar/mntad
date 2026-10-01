import os
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
import copy
import scipy.io

import pickle
import string
from string import ascii_uppercase

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.datasets import fetch_20newsgroups

ANNTHYROID_LEGACY_COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F']
ANNTHYROID_FEATURE_COLUMNS = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
ANNTHYROID_YJ_COLUMNS = ['TSH', 'T3', 'FTI']

POST_SPLIT_PREPROCESS_DATASETS = {'annthyroid', 'fault', 'glass', 'seismic', 'yeast'}

FAULT_YJ_COLUMNS = [
	'X_Minimum',
	'X_Maximum',
	'Y_Minimum',
	'Y_Maximum',
	'Pixels_Areas',
	'X_Perimeter',
	'Y_Perimeter',
	'Sum_of_Luminosity',
	'Empty_Index',
	'Outside_X_Index',
	'Luminosity_Index',
]
GLASS_YJ_COLUMNS = ['G']
YEAST_YJ_COLUMNS = [
	'Score of discriminant analysis of the amino acid content of the N-terminal region (20 residues long) of mitochondrial and non-mitochondrial proteins',
	'Score of discriminant analysis of the amino acid content of vacuolar and extracellular proteins',
	'Score of discriminant analysis of nuclear localization signals of nuclear and non-nuclear proteins',
]
SEISMIC_STANDARDIZED_DIRECT_COLUMNS = [
	'the maximum energy of the seismic bumps registered within previous shift',
]
SEISMIC_YJ_COLUMNS = [
	'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
	'a number of pulses recorded within previous shift by GMax',
	'total energy of seismic bumps registered within previous shift',
]
SYNTHETIC_YJ_COLUMNS = ['sensor_a', 'sensor_b', 'sensor_c', 'sensor_d']


def _dataset_cache_path(dataset_dir, dataset_name):
	cache_name = 'raw_data.pkl' if dataset_name in POST_SPLIT_PREPROCESS_DATASETS else 'data.pkl'
	return dataset_dir / cache_name


def _rename_annthyroid_legacy_columns(X):
	if isinstance(X, pd.DataFrame) and list(X.columns) == ANNTHYROID_LEGACY_COLUMNS:
		return X.rename(columns=dict(zip(ANNTHYROID_LEGACY_COLUMNS, ANNTHYROID_FEATURE_COLUMNS)))
	return X


class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    def __init__(self, transform=None, inv_transform=None):
        self.transform = transform if transform is not None else lambda x: x
        self.inv_transform = inv_transform if inv_transform is not None else lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), .01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_percentile = np.percentile(history, 1)
        max_percentile = np.percentile(history, 99)

        b = min_percentile - beta * (max_percentile - min_percentile)

        shifted = history - b
        a = np.percentile(shifted, alpha*100)

        if a == 0:
            a = 1
        def transform(x):
            return np.floor(1000*(x-b) / a).astype(int)
        def inv_transform(x):
            return x * a  / 1000 + b
    return Scaler(transform=transform, inv_transform=inv_transform)


# Dataset IDs for UCI/ADBench loaders; local and custom sources use None.
DATA_MAP ={
	'synthetic': None,
	'breastw':15,
	'cardio':193,
	'credit': None,
	'ecoli': 39,
	'lymphography': 63,
	'vertebral': 212,
	'wbc':17,
	'wine': 109,
	'yeast':110,
	'vifd': None,
	'fraudecom': None,
	'fakejob': None,
	'fakenews': None,
	'heart': 96,
	'arrhythmia': None, # download from https://odds.cs.stonybrook.edu/arrhythmia-dataset/
	'mulcross': None, # download from  https://www.openml.org/search?type=data&sort=runs&id=40897&status=active
	'annthyroid': 2,
	'covertype':31,
	'fault': 12,
	'glass': 14,
	'http': 16,
	'ionosphere': 18,
	'letter_recognition':20,
	'mammography': 23,
	'mulcross': None,
	'musk': 25,
	'optdigits':26,
	'pendigits':28,
	'pima':29,
	'satellite':30,
	'satimage-2':31,
	'seismic': None,
	'shuttle':32,
	'smtp':34,
	'speech':36,
	'thyroid':38,
	'vowels':40,
	'20news-0': None,
	'20news-1': None,
	'20news-2': None,
	'20news-3': None,
	'20news-4': None,
	'20news-5': None,
}


def get_direct_serialize_numerical_columns(dataset_name: str) -> List[str]:
	"""Numerical columns that should bypass MNT and be serialized with fixed precision."""
	if dataset_name == 'fault':
		return [
			'Length_of_Conveyer',
			'Steel_Plate_Thickness',
			'Minimum_of_Luminosity',
			'Maximum_of_Luminosity',
		]
	if dataset_name == 'glass':
		return [
			'C',
			'F',
		]
	if dataset_name == 'yeast':
		return [
			'Presence of HDEL substring (thought to act as a signal for retention in the endoplasmic reticulum lumen). Binary attribute',
			'Peroxisomal targeting signal in the C-terminus',
		]
	if dataset_name == 'seismic':
		return [
			'the number of seismic bumps recorded within previous shift',
			'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
			'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
			'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
			'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
			'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
			'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
			'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
			'the maximum energy of the seismic bumps registered within previous shift',
		]
	return []


def resolve_mnt_numerical_columns(X, dataset_name: str, uniform_mnt: bool = False):
	"""Return MNT and direct-serialization numerical columns for one dataset."""
	numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
	if uniform_mnt:
		return numerical_columns, []

	configured_direct_columns = set(get_direct_serialize_numerical_columns(dataset_name))
	direct_columns = [col for col in numerical_columns if col in configured_direct_columns]
	mnt_columns = [col for col in numerical_columns if col not in configured_direct_columns]
	return mnt_columns, direct_columns

def load_dataset(dataset_name, data_dir):
	dataset_dir = Path(data_dir) / dataset_name
	os.makedirs(dataset_dir, exist_ok = True)
	pkl_file = _dataset_cache_path(dataset_dir, dataset_name)
	if os.path.exists(pkl_file):
		with open(pkl_file, 'rb') as f:
			X, y= pickle.load(f)
		if dataset_name == 'annthyroid':
			X = _rename_annthyroid_legacy_columns(X)
		return X, y

	if dataset_name == 'synthetic':
		rng = np.random.default_rng(2025)
		normal = rng.normal(size=(128, 4))
		anomalies = rng.normal(loc=2.5, scale=1.2, size=(16, 4))
		X = pd.DataFrame(
			np.vstack((normal, anomalies)),
			columns=['sensor_a', 'sensor_b', 'sensor_c', 'sensor_d'],
		)
		y = np.concatenate((np.zeros(len(normal)), np.ones(len(anomalies)))).astype(np.int64)
		with open(pkl_file, 'wb') as f:
			pickle.dump((X, y), f)
		return X, y

	import adbench
	import ucimlrepo
	
	if dataset_name == 'wine':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		np_data = load_adbench_data(dataset_name)
		columns = [name.replace('_', ' ') for name in df.columns[:-1] ]

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']
	elif dataset_name == 'breastw':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		columns = [name.replace('_', ' ') for name in df.columns[1:-1] ]
		np_data = load_adbench_data(dataset_name)

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']

	elif dataset_name == 'cardio':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		L = [ k.split(' - ') for k in var_info.split('\n') ]
		column_dict = {}
		for k, v in L:
			column_dict[k] = v.strip('\r')

		df = uci_dataset.data['original']
		df = df[df['NSP'] != 2].reset_index(drop=True)
		y = df['NSP'].map({3:1, 1:0}) # map pathologic to 1, normal to 0
		y = y.to_numpy()

		df.drop(['CLASS','NSP'], inplace = True, axis = 1)
		new_columns = [ column_dict[c] for c in df.columns]
		df.columns = new_columns
		X = df 
	elif dataset_name == 'ecoli':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		columns = uci_dataset['variables']['description'][:8]
		X = uci_dataset.data['original'].drop(['class'], axis = 1)
		X.columns = columns
		X = X.drop(X.columns[0], axis=1)# drop id column
		y = uci_dataset.data['original']['class'].map({'omL':1,'imL':1,'imS':1, 'cp':0, 'im':0, 'pp':0, 'imU':0, 'om':0})
		y = y.to_numpy()
	elif dataset_name == 'lymphography':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		y = df['class'].map({1:1,2:0,3:0,4:1}) # 142 normal, 6 anomalies
		y = y.to_numpy()

		df.drop('class', inplace = True, axis = 1)
		df.drop('no. of nodes in', inplace = True, axis = 1)

		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		df['lymphatics'] = df['lymphatics'].map({1:'normal', 2:'arched', 3:'deformed', 4:'displaced'}).astype('object')
		df['defect in node'] = df['defect in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in lym'] = df['changes in lym'].map({1:'bean',2:'oval', 3:'round'}).astype('object')
		df['changes in node'] = df['changes in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in stru'] = df['changes in stru'].map({1:'no',2:'grainy', 3:'drop-like', 4:'coarse', 5:'diluted', 6: 'reticular', 7:'stripped', 8:'faint'}).astype('object')
		df['special forms'] = df['special forms'].map({1:'no',2:'chalices', 3:'vesicles'}).astype('object')
		
		for k in ['block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in', 'dislocation of', 'exclusion of no']:
			df[k] = df[k].map({1:'no',2:'yes'}).astype('object')
		
		X = df
	
	elif dataset_name == 'vertebral':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		# Treat the original Normal class as the minority anomaly class.
		df_anomaly = df[df['class'] == 'Normal']
		df_normal = df[df['class'] != 'Normal']
		df_anomaly = df_anomaly.sample(n=30, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
	
		y = df['class'].map({'Spondylolisthesis':0, 'Normal':1, 'Hernia': 0})
		y = y.to_numpy()
		df.drop('class', inplace = True, axis = 1)
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'covertype':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		for column in df.columns:
			if 'Soil' in column or 'Wilderness' in column:
				df.drop(column, axis =1 , inplace = True)
		df_normal = df[df['Cover_Type'] == 2]
		df_anomaly = df[df['Cover_Type'] == 4]
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
		
		y = df['Cover_Type'].map({2:0, 4:1})
		y = y.to_numpy()
		df.drop('Cover_Type', inplace = True, axis = 1)
		
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'fault':
		np_data = load_adbench_data(dataset_name)
		X_np, y = np_data['X'], np_data['y']
		
		columns = [
			'X_Minimum',
			'X_Maximum',
			'Y_Minimum',
			'Y_Maximum',
			'Pixels_Areas',
			'X_Perimeter',
			'Y_Perimeter',
			'Sum_of_Luminosity',
			'Minimum_of_Luminosity',
			'Maximum_of_Luminosity',
			'Length_of_Conveyer',
			'TypeOfSteel_A300',
			'TypeOfSteel_A400',
			'Steel_Plate_Thickness',
			'Edges_Index',
			'Empty_Index',
			'Square_Index',
			'Outside_X_Index',
			'Edges_X_Index',
			'Edges_Y_Index',
			'Outside_Global_Index',
			'LogOfAreas',
			'Log_X_Index',
			'Log_Y_Index',
			'Orientation_Index',
			'Luminosity_Index',
			'SigmoidOfAreas'
		]
		
		X = pd.DataFrame(data = X_np, columns = columns)
		
		# Recover categorical meanings from the standardized ADBench codes.
		steel_a300_col = columns[11]
		steel_a400_col = columns[12]
		outside_global_col = columns[20]
		
		# TypeOfSteel_A300 values: ~ -0.8168 (no) and ~ 1.2236 (yes)
		X[steel_a300_col] = pd.cut(X[steel_a300_col], bins=[-np.inf, 0, np.inf], labels=['no', 'yes']).astype('object')
		
		# TypeOfSteel_A400 values: ~ -1.2236 (no) and ~ 0.8168 (yes)
		X[steel_a400_col] = pd.cut(X[steel_a400_col], bins=[-np.inf, 0, np.inf], labels=['no', 'yes']).astype('object')
		
		# Outside_Global_Index values: ~ -1.1936 (not outside), ~ -0.1570 (partially outside), ~ 0.8796 (outside)
		X[outside_global_col] = pd.cut(
			X[outside_global_col], 
			bins=[-np.inf, -0.6, 0.3, np.inf], 
			labels=['not outside', 'partially outside', 'outside']
		).astype('object')

	elif dataset_name == 'heart':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		y = df['diagnosis'] 
		y = y.to_numpy()
		
		X = uci_dataset.data['original'].drop(['diagnosis'], axis = 1)

	elif dataset_name == 'wbc':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		df_anomaly = df[df['Diagnosis'] == 'M']
		df_normal = df[df['Diagnosis'] == 'B']
		# Preserve the benchmark's 21-anomaly subsample.
		df_anomaly = df_anomaly.sample(n=21, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
		
		y = df['Diagnosis'].map({'M':1, 'B':0})
		y = y.to_numpy()
		df.drop('Diagnosis', inplace = True, axis = 1)
		df.drop('ID', inplace = True, axis = 1)

		X = df

	elif dataset_name == 'glass':
		np_data = load_adbench_data(dataset_name)
		X_np, y = np_data['X'], np_data['y']
		X = convert_np_to_df(X_np)

	elif dataset_name == 'yeast':
		# Use the UCI labels, with ME1/ME2 as anomalies, rather than ADBench labels.
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		columns = [ s.rstrip('.') for s in uci_dataset['variables']['description'][1:9] ]
		
		y = df['localization_site'].map({'CYT':0, 'NUC':0, 'MIT':0,'ME3':0, 'ME2':1, 'ME1':1, 'EXC':0, 'VAC':0, 'POX':0, 'ERL':0}) 
		y = y.to_numpy()
		df.drop('localization_site', inplace = True, axis = 1)
		df.drop('Sequence_Name', inplace = True, axis = 1)
		df.columns = columns

		X = df

	elif dataset_name == 'vifd':
		# dataset can be downloaded from https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection/data

		df = pd.read_csv( Path(data_dir) / 'vifd'/ 'carclaims.csv')
		y = df['FraudFound'].map({"Yes":1, "No":0})
		y = y.to_numpy()

		df.drop('FraudFound', axis = 1, inplace = True)
		def split_on_uppercase(s):
			return ''.join(' ' + i if i.isupper() else i for i in s).lower().strip()
		columns = [ split_on_uppercase(c) for c in df.columns]
   
		df.columns = columns
		X = df

	elif dataset_name == 'credit':

		df = pd.read_csv(Path(data_dir) / 'credit' / 'creditcard_2023.csv')
		y = df['Class'].to_numpy()
		if 'id' in df.columns:
			df.drop('id', axis=1, inplace=True)
		df.drop('Class', axis=1, inplace=True)
		column_replacement = {
	        'Amount': 'transaction amount'
	    }
		for i in range(1, 29):
			column_replacement[f'V{i}'] = f'principal component {i}'
		df.rename(columns=column_replacement, inplace=True)
		X = df

	elif dataset_name == 'arrhythmia':
		data_path = Path(data_dir) / 'arrhythmia' / 'arrhythmia.mat'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://odds.cs.stonybrook.edu/arrhythmia-dataset/ and put it to data/arrhythmia")
			raise ValueError('arrhythmia.mat is not found in {}'.format(data_path))
		data = scipy.io.loadmat(data_path)
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)

	elif dataset_name == 'mulcross':
		data_path = Path(data_dir) / 'mulcross' / 'mulcross.arff'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://www.openml.org/search?type=data&sort=runs&id=40897&status=active and put it to data/mulcross")
			raise ValueError('mulcross.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		X = [ [x[i] for i in range(4)] for x in data]
		X_np = np.array(X)
		y = [ x[4] for x in data]
		y = [ 0 if y == b'Normal' else 1 for y in y]
		y = np.array(y)
		X = convert_np_to_df(X_np)
	elif dataset_name == 'seismic':
		# downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
		data_path = Path(data_dir) / 'seismic' / 'seismic-bumps.arff'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff and put it to data/seismic")
			raise ValueError('seismic-bumps.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		df = pd.DataFrame(data)

		column_replacement = {
			'seismic': 'result of shift seismic hazard assessment in the mine working obtained by the seismic method',
			'seismoacoustic': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method',
   			'shift': 'information about type of a shift',
			'genergy': 'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
			'gpuls': 'a number of pulses recorded within previous shift by GMax',
			'gdenergy': 'a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts',
			'gdpuls': 'a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts',
			'ghazard': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only',
			'nbumps': 'the number of seismic bumps recorded within previous shift',
			'nbumps2': 'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
			'nbumps3': 'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
			'nbumps4': 'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
			'nbumps5': 'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
			'nbumps6': 'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
			'nbumps7': 'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
			'nbumps89': 'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
			'energy': 'total energy of seismic bumps registered within previous shift',
			'maxenergy': 'the maximum energy of the seismic bumps registered within previous shift',
		}
		df.rename(columns=column_replacement, inplace=True)

		df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'] = \
			df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['information about type of a shift'] = df['information about type of a shift'].replace({b'W': 'coal-getting', b'N': 'preparation shift', 'W': 'coal-getting', 'N': 'preparation shift'})
			
		y = df['class'].map({b'0':0,b'1':1}) 
		y = y.to_numpy()

		df.drop('class', inplace = True, axis = 1)
	
		X = df
		
	elif dataset_name == 'fraudecom':
		# data downloaded from https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/data
		# preprocessing code adapted from https://www.kaggle.com/code/pa4494/catch-the-bad-guys-with-feature-engineering
		import calendar

		data_path = Path(data_dir) / 'fraudecom'
		dataset = pd.read_csv(data_path / "Fraud_Data.csv")
		IP_table = pd.read_csv(data_path / "IpAddress_to_Country.csv")

		IP_table.upper_bound_ip_address.astype("float")
		IP_table.lower_bound_ip_address.astype("float")
		dataset.ip_address.astype("float")

		def IP_to_country(ip) :
			try :
				return IP_table.country[(IP_table.lower_bound_ip_address < ip)                            
										& 
										(IP_table.upper_bound_ip_address > ip)].iloc[0]
			except IndexError :
				return "Unknown"     
			
		dataset["IP_country"] = dataset.ip_address.apply(IP_to_country)
		dataset.signup_time = pd.to_datetime(dataset.signup_time, format = '%Y-%m-%d %H:%M:%S')
		dataset.purchase_time = pd.to_datetime(dataset.purchase_time, format = '%Y-%m-%d %H:%M:%S')

		dataset["month_purchase"] = dataset.purchase_time.apply(lambda x: calendar.month_name[x.month])

		dataset["weekday_purchase"] = dataset.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])
		# Collapse singleton device IDs into a shared category.
		device_duplicates = pd.DataFrame(dataset.groupby(by = "device_id").device_id.count())
		device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
		device_duplicates.reset_index(level=0, inplace= True)

		dataset = dataset.merge(device_duplicates, on= "device_id")
		indices = dataset[dataset.freq_device == 1].index
		dataset.loc[indices, "device_id"]= "0"

		le = LabelEncoder()
		dataset['device_id'] = le.fit_transform(dataset['device_id']).astype('object')
		for column in ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'freq_device']:
			dataset.drop(column, axis=1, inplace = True)

		dataset.columns = [name.replace('_', ' ') for name in dataset.columns ]
		y = dataset['class'].to_numpy()
		X = dataset.drop("class", axis = 1)
		X = dataset.drop("device id", axis = 1)

	elif dataset_name == 'fakejob':
		# data download link: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?select=fake_job_postings.csv
		df = pd.read_csv( Path(data_dir) / 'fakejob'/ 'fake_job_postings.csv')

		df['location'].fillna('Unknown', inplace=True)
		df['department'].fillna('Unknown', inplace=True)
		df['salary_range'].fillna('Not Specified', inplace=True)
		df['employment_type'].fillna('Not Specified', inplace=True)
		df['required_experience'].fillna('Not Specified', inplace=True)
		df['required_education'].fillna('Not Specified', inplace=True)
		df['industry'].fillna('Not Specified', inplace=True)
		df['function'].fillna('Not Specified', inplace=True)
		df.drop('job_id', inplace=True, axis=1)

		text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
		df[text_columns] = df[text_columns].fillna('NaN')
		
		y = df['fraudulent'].to_numpy()
		X = df.drop('fraudulent', axis=1)
		X.columns = [name.replace('_', ' ') for name in X.columns ]
	
	
	elif dataset_name.startswith('20news-'):
		def data_generator(subsample=None, target_label=None):
			dataset = fetch_20newsgroups(subset='train')
			groups = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
				['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
				['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
				['misc.forsale'],
				['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
				['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]

			def flatten(l):
				return [item for sublist in l for item in sublist]
			label_list = dataset['target_names']
			label = []
			for _ in dataset['target']:
				_ = label_list[_]
				if _ not in flatten(groups):
					raise NotImplementedError
				
				for i, g in enumerate(groups):
					if _ in g:
						label.append(i)
						break
			label = np.array(label)
			print("Number of labels", len(label))
			idx_n = np.where(label==target_label)[0]
			idx_a = np.where(label!=target_label)[0]
			label[idx_n] = 0
			label[idx_a] = 1
			if int(subsample * 0.95) > sum(label == 0):
				pts_n = sum(label == 0)
				pts_a = int(0.05 * pts_n / 0.95)
			else:
				pts_n = int(subsample * 0.95)
				pts_a = int(subsample * 0.05)

			idx_n = np.random.choice(idx_n, pts_n, replace=False)
			idx_a = np.random.choice(idx_a, pts_a, replace=False)
			idx = np.append(idx_n, idx_a)
			np.random.shuffle(idx)

			text = [dataset['data'][i] for i in idx]
			label = label[idx]
			del dataset
	
			text = [_.strip().replace('<br />', '') for _ in text]

			print(f'number of normal samples: {sum(label==0)}, number of anomalies: {sum(label==1)}')

			return text, label
		target_label = int(dataset_name.split('-')[1])
		text, label = data_generator(subsample=10000, target_label=target_label)
		y = label
		X = pd.DataFrame(data = text, columns = ['text'])
		
	elif dataset_name == 'annthyroid':
		# ADBench annthyroid keeps the six continuous ANN-thyroid features.
		dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
		n = DATA_MAP[dataset_name]
		for npz_file in os.listdir(dataset_root):
			if npz_file.startswith(str(n) + '_'):
				print(dataset_name, npz_file)
				data = np.load(dataset_root / npz_file, allow_pickle=False)
				break
		else:
			ValueError('{} is not found.'.format(dataset_name))
		X_np, y = data['X'], data['y']
		annthyroid_columns = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
		X = pd.DataFrame(data=X_np, columns=annthyroid_columns)

	elif dataset_name in DATA_MAP.keys():
		dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
		n = DATA_MAP[dataset_name]
		for npz_file in os.listdir(dataset_root):
			if npz_file.startswith(str(n) + '_'):
				print(dataset_name, npz_file)
				data = np.load(dataset_root / npz_file, allow_pickle=False)
				break
		else: 
			ValueError('{} is not found.'.format(dataset_name))
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)
	else:
		raise ValueError('Invalid dataset name {}'.format(dataset_name))
			
	assert len(X) == len(y)

	with open(pkl_file, 'wb') as f:
		pickle.dump((X,y), f)
	
	return X, y

def load_adbench_data(dataset):
	import adbench
	dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
	if not os.path.exists(dataset_root):
		from adbench.myutils import Utils
		Utils().download_datasets(repo='jihulab')
	
	if dataset == 'cardio':
		return np.load(dataset_root / '6_cardio.npz', allow_pickle=False)

	for npz_file in os.listdir(dataset_root):
		if dataset in npz_file.lower():
			return np.load(dataset_root / npz_file, allow_pickle=False)
	else: 
		ValueError('{} is not found.'.format(dataset))

def split_data(
		X: pd.DataFrame, 
		dataset_name: str, 
		n_splits: int, 
		data_dir: str, 
		train_ratio: Optional[float] = 0.5,
		y: Optional[np.ndarray] = None,
		seed: Optional[int] = 42, 
		setting: Optional[str] = 'semi_supervised'
	) -> tuple:
	"""Return cached train/test index lists, separating caches for nondefault seeds."""
	np.random.seed(seed)
	split_name = 'split{}'.format(n_splits)
	if seed != 42:
		split_name += '_seed{}'.format(seed)
	split_dir = Path(data_dir) / dataset_name / setting / split_name
	os.makedirs(split_dir, exist_ok = True)
	
	train_indices, test_indices = [], []
	for i in range(n_splits):
		pkl_file = split_dir / 'index{}.pkl'.format(i)
		if os.path.exists(pkl_file):
			with open(pkl_file, 'rb') as f:
				train_index, test_index = pickle.load(f)
		else:
			if setting == 'unsupervised':
				normal_data_indices = np.where(y==0)[0]
				anormal_data_indices = np.where(y==1)[0]
				normal_index = np.random.permutation(normal_data_indices)
				anormal_index = np.random.permutation(anormal_data_indices)
				
				train_index = np.concatenate([normal_index[:int(train_ratio * len(normal_index))], anormal_index[:int(train_ratio * len(anormal_index))]])
				test_index = np.concatenate([normal_index[int(train_ratio * len(normal_index)):], anormal_index[int(train_ratio * len(anormal_index)):]])
			elif setting == 'semi_supervised':
				normal_data_indices = np.where(y==0)[0]
				anormal_data_indices = np.where(y==1)[0]
				data_length = len(normal_data_indices)
				index = np.random.permutation(normal_data_indices)
				
				train_index = index[:int(train_ratio * data_length)] 
				test_index = index[int(train_ratio * data_length):]
				test_index = np.concatenate([test_index, anormal_data_indices])
			else:
				raise ValueError('Invalid setting. Choose either unsupervised or semi_supervised')
			train_index = np.random.permutation(train_index)
			test_index = np.random.permutation(test_index)
			with open(pkl_file, 'wb') as f:
				pickle.dump((train_index, test_index), f)
		train_indices.append(train_index)
		test_indices.append(test_index)
	return train_indices, test_indices 

def convert_np_to_df(X_np):
	n_train, n_cols = X_np.shape
	L = list(string.ascii_uppercase) + [letter1+letter2 for letter1 in string.ascii_uppercase for letter2 in string.ascii_uppercase]
	columns = [ L[i] for i in range(n_cols) ]
	df = pd.DataFrame(data = X_np, columns = columns)
	return df


def _fit_transform_train_apply_test(X_train, X_test, columns, transformer, round_decimals=None):
	"""Fit preprocessing on training data only and reuse it for test columns."""
	columns = [col for col in columns if col in X_train.columns and col in X_test.columns]
	if len(columns) == 0:
		return

	train_values = transformer.fit_transform(X_train.loc[:, columns])
	test_values = transformer.transform(X_test.loc[:, columns])
	if round_decimals is not None:
		train_values = np.round(train_values, round_decimals)
		test_values = np.round(test_values, round_decimals)
	X_train.loc[:, columns] = train_values
	X_test.loc[:, columns] = test_values


def _apply_post_split_preprocessing(
	dataset_name,
	X_train,
	X_test,
	numerical_preprocessing='yj',
):
	"""Apply the dataset's YJ protocol, numeric standardization, or no transform."""
	if numerical_preprocessing == 'none':
		return
	if numerical_preprocessing == 'standard':
		numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			numerical_columns,
			StandardScaler(),
		)
		return
	if numerical_preprocessing != 'yj':
		raise ValueError(
			"Invalid numerical preprocessing option. Choose 'yj', 'standard', or 'none'."
		)

	if dataset_name == 'synthetic':
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			SYNTHETIC_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
		)
	elif dataset_name == 'fault':
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			FAULT_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
			round_decimals=4,
		)
	elif dataset_name == 'glass':
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			GLASS_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
		)
	elif dataset_name == 'yeast':
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			YEAST_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
		)
	elif dataset_name == 'seismic':
		direct_serialize_columns = set(get_direct_serialize_numerical_columns('seismic'))
		mnt_numerical_columns = [
			col for col in X_train.select_dtypes(include=[np.number]).columns
			if col not in direct_serialize_columns
		]
		standardized_numerical_columns = mnt_numerical_columns + [
			col for col in SEISMIC_STANDARDIZED_DIRECT_COLUMNS if col in X_train.columns
		]
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			standardized_numerical_columns,
			StandardScaler(),
		)
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			SEISMIC_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
		)
	elif dataset_name == 'annthyroid':
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			ANNTHYROID_FEATURE_COLUMNS,
			StandardScaler(),
		)
		_fit_transform_train_apply_test(
			X_train,
			X_test,
			ANNTHYROID_YJ_COLUMNS,
			PowerTransformer(method='yeo-johnson', standardize=False),
		)


def load_data(args):
	dataset_dir = Path(args.data_dir) / args.dataset
	X, y = load_dataset(args.dataset, args.data_dir)

	if 'train_ratio' not in args:
		args.train_ratio = 0.5
	if 'seed' not in args:
		args.seed = 42
	train_indices, test_indices = split_data(X, args.dataset, args.n_splits, args.data_dir, 
												args.train_ratio, y = y, seed = args.seed, setting = args.setting )
	train_index, test_index = train_indices[args.split_idx], test_indices[args.split_idx]
	X_train, X_test = X.loc[train_index].copy(), X.loc[test_index].copy()
	y_train, y_test = y[train_index], y[test_index]

	numerical_preprocessing = getattr(args, 'numerical_preprocessing', 'yj')
	_apply_post_split_preprocessing(
		args.dataset,
		X_train,
		X_test,
		numerical_preprocessing=numerical_preprocessing,
	)
	X = pd.concat([X_train, X_test], axis = 0)

	if 'binning' in args and args.binning != 'none':
		# MNT fits its own discretization during model training.
		use_mnt = getattr(args, 'use_mnt', False)
		use_numerical_embedding = getattr(args, 'use_numerical_embedding', False)
		
		if not (use_mnt or use_numerical_embedding):
			alpha = getattr(args, 'rescaling_alpha', 0.95)
			beta = getattr(args, 'rescaling_beta', 0.3)
			basic = getattr(args, 'rescaling_basic', False)
			decimals = getattr(args, 'standard_decimals', 1)
			X = normalize(X, args.binning, args.n_buckets, alpha, beta, basic, decimals)
		else:
			print(f"Skipping traditional binning ({args.binning}) because MNT ({use_mnt}) or numerical embeddings ({use_numerical_embedding}) are enabled.")
	
	if 'remove_feature_name' in args and args.remove_feature_name:
		print("Removing column names and category names.")
		L = list(ascii_uppercase) + [letter1+letter2 for letter1 in ascii_uppercase for letter2 in ascii_uppercase]
		X.columns = [ L[i] for i in range(len(X.columns))]
		
		categorical_data = X.select_dtypes(include = ['object'])
		categorical_columns = categorical_data.columns.tolist()
		le = LabelEncoder()
		for i in categorical_data.columns:
			categorical_data[i] = le.fit_transform(categorical_data[i])
		
		X_prime = X.drop(categorical_columns, axis = 1)
		X = pd.concat([X_prime, categorical_data], axis = 1)

	X_train, X_test = X.loc[train_index].copy(), X.loc[test_index].copy()
	
	return X_train, X_test, y_train, y_test

def normalize(X, method, n_buckets, alpha=0.95, beta=0.3, basic=False, decimals=1):
	"""Apply legacy non-MNT numerical representations to the supplied dataframe."""
	X = copy.deepcopy(X)
	def ordinal(n):
		if np.isnan(n):
			return 'NaN'
		n = int(n)
		if 10 <= n % 100 <= 20:
			suffix = 'th'
		else:
			suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
		return 'the ' + str(n) + suffix + ' percentile'
	
	word_list = ['Minimal', 'Slight', 'Moderate', 'Noticeable', 'Considerable', 'Significant', 'Substantial', 'Major', 'Extensive', 'Maximum']
	def get_word(n):
		n = int(n)
		if n == 10:
			return word_list[-1]
		return word_list[n]
	
	if method == 'quantile':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and  X[column].nunique() > 1:
				ranks = X[column].rank(method='min')
				X[column] = ranks / len(X[column]) * 100
				X[column] = X[column].apply(ordinal)
					
	elif method == 'equal_width':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				if X[column].nunique() > 1:
					X[column] = X[column].astype('float64')
					X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * n_buckets 
				
				if 10 % n_buckets == 0:
					X[column] = X[column].round(0) / 10
					X[column] = X[column].round(1) 
				else: 
					X[column] = X[column].round(0) / 100
					X[column] = X[column].round(2)
	elif method == 'standard':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				scaler = StandardScaler()
				scaler.fit(X[column].values.reshape(-1,1))
				X[column] = scaler.transform(X[column].values.reshape(-1,1))
				X[column] = X[column].round(decimals) 

	elif method == 'language':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
				X[column] = X[column].astype('float64')
				X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * 10
				X[column] = X[column].apply(get_word)
	
	elif method == 'quantile_rescaling':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
				col_data = X[column].values
				scaler = get_scaler(col_data, alpha=alpha, beta=beta, basic=basic)
				scaled_values = scaler.transform(col_data)
				X[column] = scaled_values
	
	else:
		raise ValueError('Invalid method. Choose either quantile, equal_width, language, standard, or quantile_rescaling')
	return X


def get_text_columns(dataset_name):
	text_columns = []
	if dataset_name == 'fakejob':
		text_columns = ['title', 'company profile', 'description', 'requirements', 'benefits']
	elif 'fakenews' == dataset_name:
		text_columns = ['title', 'text']
	elif '20news' in dataset_name:
		text_columns = ['text']
	return text_columns

def get_max_length_dict(dataset_name):
	max_length_dict = {}
	if dataset_name == 'fakejob':
		max_length_dict['title'] = 20
		text_columns = ['company profile', 'description', 'requirements', 'benefits']
		for col in text_columns:	
			max_length_dict[col] = 700
	elif 'fakenews' == dataset_name:
		max_length_dict['title'] = 30
		max_length_dict['text'] = 500
	elif '20news' in dataset_name:
		max_length_dict['text'] = 1000
	return max_length_dict
