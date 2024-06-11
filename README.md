# PhytoChat: A Multi-Turn RL-Based LLM for Diagnosis and Treatment of Plant Diseases

### Install Dependencies
```!pip install -r requirements.txt```

### Set CUDA Device(s)
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['HF_TOKEN'] = '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'
```

### Dataset Generation
1. For the Raw Dataset generation, we used two methods: (a) Website crawling using GPT-Crawler and ```Trafilatura``` Library, and (b) PDF Parsing using ```pypdfium2``` Library <br>
   a.1. Crawl data from the internet using GPT-Crawler
   * Set up a `conda` environment, install `nodejs` in it, and install dependencies.

     ```bash
     conda create -n Phytochat python=3.10
     conda activate Phytochat
     conda install -c conda-forge nodejs
     npm i
     ```

   * Edit the crawl configuration file
     ```ts
     import { Config } from "./src/config";

     export const defaultConfig: Config = {
     url: "https://www.builder.io/c/docs/developers",
     match: "https://www.builder.io/c/docs/**",
     maxPagesToCrawl: 50,
     outputFileName: "output.json",
     maxTokens: 2000000,
     };
     ```
   * Run the crawl
     ```bash
     npm start
     ```

   a.2. Crawl data from the internet using the ```Trafilatura``` Library. Check the ```crawl_webpages.py``` for the full code version. <br>
   b. Read and parse content from the PDF files using the ```parse_pdfs.py``` file. <br>

3. 
