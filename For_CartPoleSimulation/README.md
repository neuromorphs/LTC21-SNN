# Cartpolesnn

Cartpole files in their respective folders. Copy them into './CartPoleSimulation/'.

1 - 'batch' mode running virtually with the network predicting sequentially, one sample at a time.

2 - MPPI parameters read from './CartPoleSimulation/config.yml'
	'\_ControllerGUI_MPPIOptionsWindow_Spiking.py' not used for now.

3 - SNN running from './CartPoleSimulation/SNN/snn_lmu.py', uses copy of previous version of LTC21-SNN, only the inference part ("with model:" until error estimation) copied into the Predictor class.





