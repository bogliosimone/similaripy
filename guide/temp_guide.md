## Similaripy - Guide
This is a first guide, full documentation coming soon.

### Common parameters for all similarity functions
**_m1_** : the input sparse matrix for which you want calculate the similarity

**_m2_** : optional, the transpose input sparse matrix used in the calculus of the similarity, if setted to *None* just use the transpose of *m1* (default *[m2=None]*)

**_k_** : top k items per row (default *[k=100]*)

**_h_** : shrink term in normalization

**_threshold_** : all the values under this value are cutted from the final result (default *[threshold=0]*)

**_binary_** : *False* use the real values in the input matrix, *True* use binary values (0 or 1)

**_target_rows_** : if setted to *None* it compute the whole matrix otherwise it compute only the targets rows (default *[target_rows=None]*)

**_verbose_** : *True* show progress bar, *False* hide progress bar (default *[verbose=True]*)

**_format_output_** : output format for the model matrix, support values are *coo* and *csr* (default *[format_output='coo']*)

*Note: on Windows the format_output value 'csr' is not currently supported, just use the default value 'coo'*

### Similarity Algorithms

- #### Dot Product Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}%20=%20{x\cdot%20y})

- #### Cosine Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}=\frac{xy}{\left%20\|%20x%20\right%20\|\left%20\|%20y%20\right%20\|+h})

- #### Asymmetric Cosine Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}%20=%20\frac{xy}{(\sum%20x_{i}^{2})^{\alpha%20}(\sum%20y_{i}^{2})^{1-\alpha}+h})

    *&alpha;* : asymmetric coefficient *[0,1]*

- #### Jaccard Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}=\frac{xy}{\left|x\right|+\left|y\right|-xy+h})
    
- #### Dice Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}=\frac{xy}{\frac{1}{2}\left|x\right|+\frac{1}{2}\left|y\right|-xy+h})

- #### Tversky Similarity
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}=\frac{xy}{\alpha(\left|x\right|-xy)+\beta(\left|y\right|-xy)+xy+h})

    *&alpha;* : Tversky coefficient  *[0,1]*

    *&beta;* : Tversky coefficient  *[0,1]*

- #### P3&alpha; Similarity
    
    *&alpha;* : P3&alpha; coefficient

- #### RP3&beta; Similarity 

    *&alpha;* : P3&alpha; coefficient

    *&beta;* : RP3&beta; coefficient

- #### S-Plus Similarity 
    ![equation](https://latex.codecogs.com/svg.latex?\Large&space;s_{xy}=\frac{xy}{l(t_{1}(\left|x\right|-xy)+t_{2}(\left|y\right|-xy)+xy)+(1-l)(\sum%20x_{i}^{2})^{c}(\sum%20y_{i}^{2})^{1-c}+h})

    *l* : balance coefficient  *[0,1]*

    *t1* : Tversky coefficient  *[0,1]*

    *t2* : Tversky coefficient  *[0,1]*
    
    *c* : cosine coefficient  *[0,1]*