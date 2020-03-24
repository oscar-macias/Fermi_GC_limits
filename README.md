# Fermi Galactic Center Limits (arXiv:2003.10416)
Calculating Limits on WIMP DM annihilation cross sections from Femi GC gamma ray data.
This repository contains the data files for Figure 1, but it also contains the full
log-like data files and our Bayesian pipeline necessary to reproduce
the figures in our paper. So, if you are interested in including our derived limits in your plot please 
go to directory Fig1_data, but if you are interested in constraining your own model
or simply reproducing our results, please see below instructions.

To reproduce our DM limits, you can run:

```console
python Fermi_GC_limits.py --channel=0 --GDE_model=0 --DM_model=0 --ignore
```
The above line of code will produce the 95% CL upper limits in on $<\sigma v>$ for our standard GDE model 
and DM profile given by an NFW with gamma=1.2. More details about the options supported are given next.

``` console
command line args: channel - an integer index specifying which annihilation to calculate for; 0:bbar, 1:tau, 2:mu, 3:W, 4:Z , 5:higgs
                   GDE_model - an integer index specifying the kind of GDE model (or whether to vary NFW gamma); 0:GDE_baseline, 1:GDE_Dust, 2:GDE_Gas 3:NFW_gamma, 4:GDE_CSE, 5:GDE_ICS_6rings
                   DM_model - an integer index specifying which DM morphology is used; 0:NFW, 1:triaxial NFW, 2:Read, 3:triaxial Read (if kind is 3:NFW_gamma, this index runs from 0-9 spanning gamma= {0.5,1.5}, skipping 1.2 i.e the baseline)
                   trunc - an integer specifying the number of low-energy data points to ignore, truncating from the lowest bin
                   ignore - Boolean flag to ignore ''divide by zero'' errors encounterd in ''log''
```

Note: The limits shown in Fig.1 of arXiv:2003.10416 are the weakest limits of many different ones 
that considered different Galactic diffuse emission (GDE) models and dark matter (DM) morphologies.
In practice you should run this script a few more times using different GDE models and DM morphologies
to get exactly our Fig.1. However, the individual limits are shown in our Fig.2, and you can directly 
compare the result of this calculation to that one.

## Want to know more about how our Bayesian pipeline actually works?

We have constructed a script that uses our Bayesian method to reproduce the results in <https://arxiv.org/pdf/1503.02641.pdf>.
In particular, this script reads in the log-like data for the Draco dSph and dumps out the 95% CL upper limits on $<\sigma v>$
as a function of DM mass, but can be easily expanded to reproduce the full results in the paper above.

To run the code execute:  
```console
python draco_test.py --channel=0 --ignore
```
