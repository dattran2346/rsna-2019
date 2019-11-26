import os
from config import RSNAConfig
import pandas as pd

if __name__ == "__main__":
    rsna = RSNAConfig()
    images = []
    for rdir,_,files in os.walk(rsna.conf.stage2_test_dir):
        for file in files:
            if '.dcm' in file:
                filename, file_extension = os.path.splitext(file)
                images.append(filename)
    df = pd.DataFrame(images, columns = ['image'])
    df.to_csv(rsna.conf.testset_stage2, index=False)