The extracellular matrix is a complex substance found within the confines of the intercellular tissue space. Besides providing anchoring support, the extracellular matrix, secreted and assembled by resident cells, is a mechanical and biochemical environment that directs cellular functions and processes via a collection of various stimuli, prompting gene expression profiles to reflect developmental and physiological contexts. Being a dynamic structure, the extracellular matrix composition is subject to change as a function of pathology and age. Presently, there is a lack of a unified consensual understanding of the qualitative and quantitative aspects of said age-related compositional changes on a tissue/organ level. Herein, I describe a database that aggregates published proteomic datasets on extracellular matrix aging signatures with a tissue-level granularity, allowing direct comparison of matrisomal protein abundances across different compartments and methodologies.

The very first iteration of the database will be a CSV file, containing the information on protein abundances and accompanying data in a standardized format. For example,

Protein\_ID,Protein\_Name,Gene\_Symbol,Tissue,Species,Age,Age\_Unit,Abundance,Abundance\_Unit,Method,Study\_ID,Sample\_ID
P02452,Collagen alpha-1(I) chain,COL1A1,Skin,Homo sapiens,25,years,1500,ppm,LC-MS/MS,PMID:12345678,S001

For this, I started expanding my [dataset inventory](https://docs.google.com/spreadsheets/d/1JSV8jQSin9vTu8mYVX0j-lZEAF2fyrfpiU5-K8xS3t0/edit?usp=sharing) with the aim of populating it with as much useful information as possible from the selected papers and associated supplementary data, paying extra attention to dataset structure and content; e.g. sampled age bins (they may be different between studies), how protein abundances are reported (they may vary), how protein IDs are reported (i.e protein vs. gene nomenclature) – all to determine common and contrasting features of the datasets in order to strategize standardization. In addition, to get a better “feel” of the data to account for any limitations, I’ve surveyed methodologies used in each of the papers, compiling workflows in Obsidian for better cross-referencing.

In case the abundance measures differ across datasets (which they do), a normalization method will have to be applied; e.g. conversion to percentiles or *z*\-scores. Once I have numerical columns with normalized abundance data, I will use [Matrisome AnalyzeR](https://matrinet.shinyapps.io/MatrisomeAnalyzer/) – a R package (and a web tool) – to classify and annotate my dataset. Database design is the next step.

One of the suggestions under my pre-registration was that I look into existing metadata ontologies, such as [coderdata](https://github.com/PNNL-CompBio/coderdata) (PNNL), [CELLxGENE](https://cellxgene.cziscience.com/) (CZI), or [AnVIL](https://anvilproject.org/) (NIH). Using one of the existing standards would make sharing data and collaboration easier, as well as ensure interoperability with other databases and tools. I started looking at the documentation for these standards.

To sum up, I’m at the data organization stage with the following tasks:
\- Compile datasets into a standardized format;
\- Ensure consistent protein naming/identifiers across datasets;
\- Create metadata for each dataset (tissue type, age, methodology, etc.).
\- Normalize protein abundance values;
\- Annotate the resulting dataset using the matrisome-tailored toolkit.
