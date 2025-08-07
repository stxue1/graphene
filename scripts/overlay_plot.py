import argparse
import os
import sys
from pathlib import Path
from typing import List, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.figure import figaspect
from scipy.stats import norm

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*")
    options=parser.parse_args(args)
    # temp
    images = options.images
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0])
    # plt.imshow(images[1], alpha=0.5)
    from pypdf import PdfReader, PdfWriter

    with open(images[0], 'rb') as file1, open(images[1], 'rb') as file2:
        reader1 = PdfReader(file1)
        reader2 = PdfReader(file2)

        writer = PdfWriter()

        num_pages = min(len(reader1.pages), len(reader2.pages))

        for i in range(num_pages):
            page1 = reader1.pages[i]
            page2 = reader2.pages[i]

            page1.merge_page(page2)

            writer.add_page(page1)

        with open('output_overlay.pdf', 'wb') as output_pdf:
            writer.write(output_pdf)

    print("Overlay completed. Output saved as 'output_overlay.pdf'.")

if __name__ == "__main__":
    main()