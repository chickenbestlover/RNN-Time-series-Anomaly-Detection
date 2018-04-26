import requests
import os
from pathlib import Path
import pickle
from shutil import unpack_archive

urls = dict()
urls['ecg']=['http://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip',
             'http://www.cs.ucr.edu/~eamonn/discords/mitdbx_mitdbx_108.txt',
             'http://www.cs.ucr.edu/~eamonn/discords/qtdbsele0606.txt',
             'http://www.cs.ucr.edu/~eamonn/discords/chfdbchf15.txt',
             'http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt']
urls['gesture']=['http://www.cs.ucr.edu/~eamonn/discords/ann_gun_CentroidA']
urls['space_shuttle']=['http://www.cs.ucr.edu/~eamonn/discords/TEK16.txt',
                       'http://www.cs.ucr.edu/~eamonn/discords/TEK17.txt',
                       'http://www.cs.ucr.edu/~eamonn/discords/TEK14.txt']
urls['respiration']=['http://www.cs.ucr.edu/~eamonn/discords/nprs44.txt',
                     'http://www.cs.ucr.edu/~eamonn/discords/nprs43.txt']
urls['power_demand']=['http://www.cs.ucr.edu/~eamonn/discords/power_data.txt']

for dataname in urls:
    raw_dir = Path('dataset', dataname, 'raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in urls[dataname]:
        filename = raw_dir.joinpath(Path(url).name)
        print('Downloading', url)
        resp =requests.get(url)
        filename.write_bytes(resp.content)
        if filename.suffix=='':
            filename.rename(filename.with_suffix('.txt'))
        print('Saving to', filename.with_suffix('.txt'))
        if filename.suffix=='.zip':
            print('Extracting to', filename)
            unpack_archive(str(filename), extract_dir=str(raw_dir))



    for filepath in raw_dir.glob('*.txt'):
        with open(str(filepath)) as f:
            # Label anomaly points as 1 in the dataset
            labeled_data=[]
            for i, line in enumerate(f):
                tokens = [float(token) for token in line.split()]
                if raw_dir.parent.name== 'ecg':
                    # Remove time-step channel
                    tokens.pop(0)
                if filepath.name == 'chfdbchf15.txt':
                    tokens.append(1.0) if 2250 < i < 2400 else tokens.append(0.0)
                elif filepath.name == 'xmitdb_x108_0.txt':
                    tokens.append(1.0) if 4020 < i < 4400 else tokens.append(0.0)
                elif filepath.name == 'mitdb__100_180.txt':
                    tokens.append(1.0) if 1800 < i < 1990 else tokens.append(0.0)
                elif filepath.name == 'chfdb_chf01_275.txt':
                    tokens.append(1.0) if 2330 < i < 2600 else tokens.append(0.0)
                elif filepath.name == 'ltstdb_20221_43.txt':
                    tokens.append(1.0) if 650 < i < 780 else tokens.append(0.0)
                elif filepath.name == 'ltstdb_20321_240.txt':
                    tokens.append(1.0) if 710 < i < 850 else tokens.append(0.0)
                elif filepath.name == 'chfdb_chf13_45590.txt':
                    tokens.append(1.0) if 2800 < i < 2960 else tokens.append(0.0)
                elif filepath.name == 'stdb_308_0.txt':
                    tokens.append(1.0) if 2290 < i < 2550 else tokens.append(0.0)
                elif filepath.name == 'qtdbsel102.txt':
                    tokens.append(1.0) if 4230 < i < 4430 else tokens.append(0.0)
                elif filepath.name == 'ann_gun_CentroidA.txt':
                    tokens.append(1.0) if 2070 < i < 2810 else tokens.append(0.0)
                elif filepath.name == 'TEK16.txt':
                    tokens.append(1.0) if 4270 < i < 4370 else tokens.append(0.0)
                elif filepath.name == 'TEK17.txt':
                    tokens.append(1.0) if 2100 < i < 2145 else tokens.append(0.0)
                elif filepath.name == 'TEK14.txt':
                    tokens.append(1.0) if 1100 < i < 1200 or 1455 < i < 1955 else tokens.append(0.0)
                elif filepath.name == 'nprs44.txt':
                    tokens.append(1.0) if 16192 < i < 16638 or 20457 < i < 20911 else tokens.append(0.0)
                elif filepath.name == 'nprs43.txt':
                    tokens.append(1.0) if 12929 < i < 13432 or 14877 < i < 15086 or 15729 < i < 15924 else tokens.append(0.0)
                elif filepath.name == 'power_data.txt':
                    tokens.append(1.0) if 8254 < i < 8998 or 11348 < i < 12143 or 33883 < i < 34601 else tokens.append(0.0)
                labeled_data.append(tokens)

            # Fill in the point where there is no signal value.
            if filepath.name == 'ann_gun_CentroidA.txt':
                for i, datapoint in enumerate(labeled_data):
                    for j,channel in enumerate(datapoint[:-1]):
                        if channel == 0.0:
                            labeled_data[i][j] = 0.5 * labeled_data[i - 1][j] + 0.5 * labeled_data[i + 1][j]

            # Save the labeled dataset as .pkl extension
            labeled_whole_dir = raw_dir.parent.joinpath('labeled', 'whole')
            labeled_whole_dir.mkdir(parents=True, exist_ok=True)
            with open(str(labeled_whole_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                pickle.dump(labeled_data, pkl)

            # Divide the labeled dataset into trainset and testset, then save them
            labeled_train_dir = raw_dir.parent.joinpath('labeled','train')
            labeled_train_dir.mkdir(parents=True,exist_ok=True)
            labeled_test_dir = raw_dir.parent.joinpath('labeled','test')
            labeled_test_dir.mkdir(parents=True,exist_ok=True)
            if filepath.name == 'chfdb_chf13_45590.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[:2439], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2439:3726], pkl)
            elif filepath.name == 'chfdb_chf01_275.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[:1833], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[1833:3674], pkl)
            elif filepath.name == 'chfdbchf15.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[3381:14244], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[33:3381], pkl)
            elif filepath.name == 'qtdbsel102.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[10093:44828], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[211:10093], pkl)
            elif filepath.name == 'mitdb__100_180.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2328:5271], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[73:2328], pkl)
            elif filepath.name == 'stdb_308_0.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2986:5359], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[265:2986], pkl)
            elif filepath.name == 'ltstdb_20321_240.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[1520:3531], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[73:1520], pkl)
            elif filepath.name == 'xmitdb_x108_0.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[424:3576], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[3576:5332], pkl)
            elif filepath.name == 'ltstdb_20221_43.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[1121:3731], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[0:1121], pkl)
            elif filepath.name == 'ann_gun_CentroidA.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[3000:], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[:3000], pkl)
            elif filepath.name == 'nprs44.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[363:12955], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[12955:24082], pkl)
            elif filepath.name == 'nprs43.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[4285:10498], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[10498:17909], pkl)
            elif filepath.name == 'power_data.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[15287:33432], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[501:15287], pkl)
            elif filepath.name == 'TEK17.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2469:4588], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[1543:2469], pkl)
            elif filepath.name == 'TEK16.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[521:3588], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[3588:4539], pkl)
            elif filepath.name == 'TEK14.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2089:4098], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[97:2089], pkl)

nyc_taxi_raw_path = Path('dataset/nyc_taxi/raw/nyc_taxi.csv')
labeled_data = []
with open(str(nyc_taxi_raw_path),'r') as f:
    for i, line in enumerate(f):
        tokens = [float(token) for token in line.strip().split(',')[1:]]
        tokens.append(1) if 150 < i < 250 or   \
                            5970 < i < 6050 or \
                            8500 < i < 8650 or \
                            8750 < i < 8890 or \
                            10000 < i < 10200 or \
                            14700 < i < 14800 \
                          else tokens.append(0)
        labeled_data.append(tokens)
nyc_taxi_train_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','train',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_train_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_train_path),'wb') as pkl:
    pickle.dump(labeled_data[:13104], pkl)

nyc_taxi_test_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','test',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_test_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_test_path),'wb') as pkl:
    pickle.dump(labeled_data[13104:], pkl)