### TextAugmentation-CLONALG-AMR
This repository showcases a graph-based approach and the Clonal Selection Algorithm (CLONALG) for text augmentation in Natural Language Processing (NLP) tasks.

## Overview
Annotated data plays a crucial role in training machine learning models. However, manually labeling large amounts of data with high-quality annotations can be time-consuming and labor-intensive. In the field of NLP, the labels provided by human annotators vary in competency, training, and experience, leading to arbitrary and ambiguous standards. To address the challenges of insufficient high-quality labels, researchers have been exploring automated methods for enhancing training and testing datasets.

In this paper, we present a novel method that leverages the Clonal Selection Algorithm (CLONALG) and abstract meaning representation (AMR) graphs to improve the quality and quantity of data in two cybersecurity problems: fake news identification and sensitive data leak detection. Our proposed approach demonstrates significant enhancements in dataset performance and classification accuracy, surpassing baseline results by at least 5%.

## Repository Structure
This repository contains the following files and directories:

- data/: This directory contains the dataset files used in the experiments.
- src/: This directory contains AMR distance Metrics Code.
- code/: This directory contains the implementation of the TextAugmentation-CLONALG-AMR method.
- results/: This directory stores the results obtained from applying the method on the datasets.
- README.md: This file provides an overview of the repository and the research paper.
- LICENSE: This file contains the licensing information for the repository.

## Usage
To utilize the TextAugmentation-CLONALG-AMR method for text augmentation, follow these steps:

1. Clone this repository to your local machine.
2. Install the necessary dependencies mentioned in the requirements file.
3. Create a `data` folder in the root directory and place your dataset CSV file inside it. Make sure the column name containing the text data is named 'text' in the CSV file.
4. Open the `main.py` file and modify the `input_file` variable to specify the path to your CSV file.
5. Run the `main.py` script.
6. The augmented data will be saved as a new CSV file in the `results` directory with the same name as the input file but with "_augmented" appended to it.

## Example

To run the text augmentation on your own CSV file:

1. Create a `data` and `results` folder in the root directory of the repository.
2. Place your dataset CSV file inside the `data` folder.
3. Open the `main.py` file and modify the `input_file` variable:

   ```python
   input_file = "data/your_file.csv"  # Specify the path to your CSV file
    ```

## License
to be continued