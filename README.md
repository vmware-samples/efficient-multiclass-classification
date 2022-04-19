# Duet scikit classifier (v1.0)


## Overview

Duet is a decision tree ensemble method based multiclass classification 
framework that offers a more efficient resource usage while preserving and even 
improving the classification accuracy in comparison to standard monolithic 
models.

Duet is based on a small bagging ensemble model and a booting model.<br/>
The current implementation of Duet is based on Random Forest and XGBoost.

## Documentation

More details about the Duet can be found in the following paper:<br/>
"Efficient Multiclass Classification with Duet" [EuroMLSys '22]<br/>
<https://dl.acm.org/doi/abs/10.1145/3517207.3526970><br/>
<https://euromlsys.eu/pdf/euromlsys22-final4.pdf>

## Files:

#### duet_classifier.py 
Duet scikit classifier

#### classification_example.py
Basic classification example by Duet

#### grid_search_example.py
Basic grid search example with Duet

## Prerequisities:
numpy<br/>
pandas<br/>
skleran<br/>
xgboost<br/>


or alternatively, run:<br/>
$ pip3 install -r requirements.txt

## Contributing

The efficient-multiclass-classification project team welcomes contributions from the community. Before you start working with efficient-multiclass-classification, please
read our [Developer Certificate of Origin](https://cla.vmware.com/dco). All contributions to this repository must be
signed as described on that page. Your signature certifies that you wrote the patch or have the right to pass it on
as an open-source patch. For more detailed information, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3 License 

## Contact us

For more information, support and advanced examples contact:<br/>
Yaniv Ben-Itzhak, [ybenitzhak@vmware.com](mailto:ybenitzhak@vmware.com)<br/>
Shay Vargaftik, [shayv@vmware.com](mailto:shayv@vmware.com)<br/>
