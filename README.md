# _In vivo_ Delineation of a Body-Part-Specific Topography in Mouse Secondary Motor Cortex



This project is organized into two repositories containing the analysis scripts and data accompanying the manuscript.

Code Repository (this one): [https://github.com/hillmanlab/M2-topography](https://github.com/hillmanlab/M2-topography)

Data Repository: [https://doi.org/10.25452/figshare.plus.28633379](https://doi.org/10.25452/figshare.plus.28633379)

All scripts are organized into clear folders by their specific analysis or figure, facilitating reproducibility and ease of navigation.

## Hardware Requirements

- **OS:** Analyses were performed on CentOS Linux.
    
- **CPU/RAM:** The primary environment used servers with **2× Intel Xeon Gold 6140 18-core processors (36 cores total)** and **192 GB RAM**.
    > A multi-core CPU and substantial memory are recommended for reproducing the analyses.



## Code Repository

_______________________________________

### `make_0_prepare`

Scripts for preprocessing and generating datasets used throughout the analysis pipeline.

- **common-functions**  
    General-purpose utility functions used across analyses and figure generation.
    
- **config-colormap**  
    Generates consistent colormaps for manuscript figures.
    
- **registration**  
    Performs inter-mouse registration using landmark points (used in Figure 1).
    
- **sensory-boop-test**  
    Identifies sensory cortical regions for specific body parts, derived from sensory mapping under dexmedetomidine sedation (used in Figure 1).
    
- **extract epochs**  
    Scripts for extracting epoch-specific neural and behavioral data:
    
    - **extract-rest**: Resting-state epochs, utilized for constrained least squares (CLS) forward pass (Figures 7, 8, S5 and S7).
    - **extract-rest-to-running-epochs**: Rest-to-running epochs (Figure 3 and S1).
    - **extract-running-epochs**: Epochs from running behaviors (Figure 8).
    - **extract-CSN-epochs**: Locomotion epochs for CSNs analysis (Figure 6).
    - **extract-lick-epochs**: Licking behavior epochs (Figure 8).
    - **extract-integrated-motif**: Integrated movement motif epochs(Figure 8).
    - **extract-stimulus-epochs**: Stimulus-evoked wheel running epochs (Figure S1).
- **heterogeneity analysis**
    
    - **heterogeneity-extract**: Extraction and calculation of pan-cortical resting-state functional connectivity patterns (used in Figure 4, Figure S3, Movie S3).
    - **contour-S1-M2-topography**: Extracts contours defining M2 and S1 regions from pan-cortical heterogeneity patterns (used in Figure 4, Figure S3, Movie S3).
- **modeling**  
    Scripts for 1D-CNN and linear modeling analysis (Figure S2).
    
- **refine-M2-ROIs**  
    Scripts for manually refining region-of-interest (ROI) boundaries in M2.
    
- **behavior-syllables** <br>
	Saves all behavioral signals into pandas DataFrame (Figure 2)
	
- **CNN&linear** <br>
	Decodes behavioral signals using neural activity (Figure S2)
	
- **reconstruct-500-to-full**  
	load full-resolution (256 x 256 pixels) data (Note: Data size is large; ensure sufficient memory is available). 
	
- **CSN-M1-prelude** <br>
	Aggregated analysis for simultaneous recordings of Thy1 and CSNs activity across movement types (Figure 6 and S5) 
	
- **CLS-all-mice** <br>
	Spatiotemporal un-mixing analysis using constrained least squares (CLS) (Figure 7, 8, and S6)

### `make_1_figures`

Scripts for generating main and supplementary figures, organized by figure number (e.g., `figure_X`).

### `make_2_movies`

Scripts for generating supplementary videos, organized by movie number (e.g., `movie_X`).


_______________________________________

_______________________________________
<br>
<br>
<br>
<br>



## Data Repository Structure


Download the full data repository from: [https://doi.org/10.25452/figshare.plus.28633379](https://doi.org/10.25452/figshare.plus.28633379)

Unzip to your local directory (e.g.,  
`~/data_submission/steps_data` and `~/data_submission/prepares_data`), which will serve as the **raw data root** and **analysis root**, respectively.


_______________________________________

### `steps_data`

Includes synchronized neural activity, hemodynamic signals, LEDs illumination data collected via wide-field optical mapping (WFOM).

- **step_0**: Configuration files
    
    - `cmXXX` (Mouse-specific configuration files, including `IDX_sorted` and cortical `mask`).
        
- **step_1**: Full-resolution brain imaging data, stored as compact SVD-decomposed `.mat` files in hdf5 format
    
    - `cmXXX`
        
        - `1` (Experimental session/day number)
            
            Individual `.mat` files for different sequential 10-minute recordings (e.g., runB, runC, ...), along with corresponding preview movies. Each data file contains: <br>
            **-m**: acquisition information.<br>
            **-mask**: Boolean mask used to remove non-cortical areas outside the dorsal cortical imaging window. <br>
            -**spatial (`C`) and temporal (`S`) SVD components** for <br>
	            -Raw LED illumination data (`lime`, `green`, `red`; `blue` is included only in 2-color imaging sessions)<br>
	            -Neuronal calcium-dependent fluorescence signals: `jrgeco` (all mice) and `gcamp` (only in 2-color mice)<br>
	            -Hemodynamic signals: changes in oxyhemoglobin (`chbo`), total oxyhemoglobin (`chbt`). Changes in de-oxyhemoglobin (`chbr`) computed as `chbt` - `chbo` <br>
	        For reconstructing original data matrices from SVD components, refer to the code provided below.
                
- **step_2**: Parcellated brain data (500 k-means clustered regions).
    
    - `cmXXX`
        
        - `1` (Experimental session/day number)
            
            Extracted time courses for 500 regions and associated preview movies.
                
- **step_3**: Behavioral signals extracted from webcam footage.
    
    - `cmXXX`
        
        - `1` (Experimental session/day number)
            
            Summary of motion energy (face and body regions) extracted from webcam videos.
                
### `figures_data`

- Support data for making main, supplementary figures and movies, organized by figure number (e.g., `figure_X`).
    

### `webcams_data`

- **DLC-output**: Webcam videos labeled via DeepLabCut pose estimation.
    

### `prepares_data`

- **config**: Configuration support files (pickle or hdf5 mat format, including color palettes, colormaps, brain masks, etc.).
    
- **registration**: Data used for inter-mouse registration (Figure 1).
    
- **sensory-boop-test**: Contours of sensory regions (Figure 1).
    
- **behavior-segment**: Pickle files containing continuous and discrete behavioral data (pandas DataFrame format; Figure 2, Movies S1-S2).
    
- **epochs-rest-to-running**: Used in Figure 3.
    
- **epochs-csn**: Used in Figure 6, S5, Movies S5, S6.
    
- **epochs-rest**: Used for CLS forward pass (Figure 7, 8, S6 and S7).
    
- **epochs-running**: Used in Figure 8; Movie S7.    
     
- **epochs-licking**: Used in Figure 8; Movie S8.
      
- **epochs-integrated**: Used in Figure 8; Movie S9.
    
- **heterogeneity**: resting-state heterogeneity patterns data (Figure 4, Figure S3, Movie S3).
    
- **heterogeneity-contours**: Contours extracted from pan-cortical heterogeneity patterns (Figure 4, Figure S3).
    
- **index-M2**, **index-S1**, **index-M1**, **index-extra**: region IDX indices for M2, S1, M1 and extra regions.
    
- **modeling**: Data for 1D-CNN and linear modeling (Figure S2).
    
- **CLS**: Spatial templates fitted using constrained least squares (Figures 7-8, Figure S6-7, Movies S7-S9).
    


## Mouse Data Details

Data from multiple mice are included, organized by mouse identifier (`cmXXX`). Mice cm125, cm126, cm128, and cm193 are Thy1-jRGECO1a mice imaged using single-color WFOM. Mouse cm229 is a Thy1-jRGECO1a mouse with corticospinal neurons projecting to the right forelimb labeled with GCaMP7f, imaged using two-color WFOM.

## Data File Formats and Usage


### Loading step_1 Data (hdf5 .mat format)

Each `.mat` file includes SVD spatial component `C`, SVD temporal component `S`, acquisition info `m`, and masks for LED and fluorescence data (`jrgeco` or `gcamp`).

Example codes to reconstruct the original data matrix:

```
import numpy as np
import h5py

filename = "~/steps_data/step_1/cmXXX/2/runC_stim_1.mat"

# Load jrgeco data
with h5py.File(filename, 'r') as f_full:
	m = f_full['m']
	C = np.array(f_full['C']['jrgeco'])
	S = np.array(f_full['S']['jrgeco'])
	nanidx = np.squeeze(m['nanidx']) == 1
	CS = np.dot(S, C).transpose()
	jrgeco = np.full((len(nanidx), CS.shape[-1]), np.nan)
	jrgeco[nanidx, :] = CS
	jrgeco = jrgeco.reshape((256, 256, -1)).transpose((1, 0, 2))
```

```
# Reconstruct raw LED illumination data (lime channel example)
with h5py.File(filename, 'r') as f_full:
	C = np.array(f_full['C']['lime'])
	S = np.array(f_full['S']['lime'])
	CS = np.dot(S, C).transpose()
	lime = CS.reshape((256, 256, -1)).transpose((1, 0, 2))
```

### Loading Pickle Files

```
import pickle
with open(pkl_filename, 'rb') as file:
	pickle_model = pickle.load(file)
```

### Usage of step_2 Data

Pixels from full-resolution images were clustered into 500 regions using k-means clustering (run individually for each mouse). Region indices (`IDX_sorted`) are ordered systematically from right-to-left and anterior-to-posterior. Each mouse’s corresponding `IDX_sorted` index is stored in the **step_0** folder and applied to the **step_1** data to extract neural activity time courses (see the figure below). This reduces data dimensionality from 256 × 256 pixels to 500 spatially clustered regions, facilitating efficient downstream analyses.

![](IDX_sorted_demo.png)


