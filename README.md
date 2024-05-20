# PhytoChat

### Crawl sites using GPT-Crawler

Set up a `conda` environment, install `nodejs` in it, and install dependencies.

```bash
conda create -n Phytochat python=3.10
conda activate Phytochat
conda install -c conda-forge nodejs
npm i
```

Edit the crawl configuration file

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

Run the crawl

```bash
npm start
```
