from vggsfm.utils.data_processor import LINEMOD_OneposeProcessor
import traceback
cat_list = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone', 
            'cup', 'bowl']
# cat_list = ['ape']
def main():
    for cat in cat_list:
        data_path = f'data/linemod_onepose'
        output_path = f'data/linemod_onepose/vggsfm_prepared/lm_test_{cat}'
        try:
            processor = LINEMOD_OneposeProcessor(data_path, output_path, length=None, stride=1, catogoery=cat)
            processor.process(split='train')
            # processor.process(split='test')
        except Exception as e:
            print(f"Error in processing {cat}")
            print(e)
            traceback.print_exc()

    
if __name__ == '__main__':
    main()