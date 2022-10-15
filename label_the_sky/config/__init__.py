import yaml

BANDS = ['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']
#CLASS_MAP = {'GALAXY': 0, 'STAR': 1, 'QSO': 2, 'UNKNOWN': 3}
CLASS_MAP = {2: 0, 1: 1, 0: 2, 'UNKNOWN': 3} # 0 in the df is QSO (which is 2 in the program), etc... 

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)