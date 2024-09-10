from vggsfm.utils.data_processor import LINEMODProcessor

cat_list = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone', 
            'cup', 'bowl']

def main():
    for cat in cat_list:
        data_path = f'/home/SSD2T/yyh/dataset/lm/test'
        output_path = f'/home/yyh/lab/vggsfm/data/lm_test_{cat}'
        
        processor = LINEMODProcessor(data_path, output_path, length=20, stride=2, catogoery=cat)
        processor.process()
    
if __name__ == '__main__':
    main()