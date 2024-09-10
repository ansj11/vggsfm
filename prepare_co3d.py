from vggsfm.utils.data_processor import CO3DProcessor

def main():
    data_path = '/home/SSD2T/yyh/dataset/co3d_test_raw'
    
    import os
    for cat in os.listdir(data_path):
        cat_path = os.path.join(data_path, cat)
        
        first_seq = os.listdir(cat_path)
        # sort the sequence by name
        first_seq.sort()
        first_seq = first_seq[0]
        output_path = '/home/yyh/lab/vggsfm/data/co3d_test_' + cat
        processor = CO3DProcessor(data_path, output_path, length=50, stride=5, sequence_name=first_seq, catogoery=cat)
        processor.process()
    
if __name__ == '__main__':
    main()
            