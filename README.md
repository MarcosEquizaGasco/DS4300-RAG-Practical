# DS4300-RAG-Practical
### Jacob Gottesman and Marcos Equiza Gasco
### Team jacob_marcos
### 04/02/2025

This repository contains all needed code to run the second practical of DS 4300.

# Instructions for Using Repo

## Creating a DB Collection and Using the RAG System

First ensure that you have created a Docker Container for the database you want to use, and make sure you are running it.

To process all the data, split it into chunks, and index it into the database, use the file `pipeline_test.py`. First, determine the file location of the folder containing the data you want to use. Specify the database, LLM, embedding model, chunk size and overlap you want to use. 

Open a terminal at the root of this repository and run 

` python pipeline_test.py `

The interactive search will load into the terminal, and will prompt the user for a query. After writing a query, a response will be returned.

## Running the Experiment

Run the file `experimentation.ipynb` to test all combinations of databases, embedding models, chunk size, overlap, and clean text used in our experiment. This file automatically creates collections in the database and tests them with the different parameters. This will automatically create a .csv file called `results.csv` with the data from the experiment. 

## Recreating Visualizations From the Report

Run the file `results_analysis.ipynb` to recreate all visualization and tables about the experiment used within the report.
