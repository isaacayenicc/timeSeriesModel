# Source access log

Access date: 2026-10-30 UTC. Inherited proxy and normal TLS verification were preserved. Publisher/Crossref/MathWorks/arXiv destinations remain policy-denied. GitHub source repositories and GitHub public repository search were accessed normally; no blocked-host proxy, mirror gateway, credential extraction or direct-route bypass was used. Saving academic-domain additions in a draft does not establish access.

Discovery used public repository search, followed by original author/lab READMEs, an author publication listing, and lightweight Git-tree inspection. No article manuscript was found in the inspected author trees. Source attribution and a repository name were checked rather than assumed: an unrelated MCKD-acronym archive contained molecular-modeling material and was excluded. A search hit or a paper title is not evidence its article was read.

The snapshots in `evidence/` preserve the exact bytes consulted. Mutable upstream branch URLs are identified by SHA-256 below; installed package versions are separately frozen by the project. Source files were read at relevant documented functions rather than claimed read line by line. Some related author sources disagree on online/issue dates (R15–R18); the matrix and BibTeX notes preserve those discrepancies rather than invent publisher verification.

| Key | URL | Status | SHA-256 / error |
|---|---|---|---|
| tcn_readme | https://raw.githubusercontent.com/locuslab/TCN/master/README.md | retrieved | 346b345ff451b0e3069c17e6c59718cfd5195b48974c0bf518ae3b9fd1cc1f77 |
| gru_groundhog | https://raw.githubusercontent.com/lisa-groundhog/GroundHog/master/README.md | retrieved | 026f4de0b9b18706f8d1804377275fbbf98a94098fea055fcc89063d508b227c |
| gru_dl4mt | https://raw.githubusercontent.com/nyu-dl/dl4mt-tutorial/master/README.md | retrieved | 80f8bcad9841d14d0e9116c3b4d9d7883bb326320f70fcc993be457b1066084d |
| scipy_kurtosis | https://raw.githubusercontent.com/scipy/scipy/main/scipy/stats/_stats_py.py | retrieved | 98f44303bdfdb7ad09edc5c2f9ea6264cb49f22ad04ea31685ade069f77f813f |
| scipy_hilbert | https://raw.githubusercontent.com/scipy/scipy/main/scipy/signal/_signaltools.py | retrieved | 85be0fa304bd86261fadcec817ccfa36a89ef03090bbbcb681c79d4eaf1a5ab4 |
| scipy_spectral | https://raw.githubusercontent.com/scipy/scipy/main/scipy/signal/_spectral_py.py | retrieved | dc3d6cefbdc08902d135b85e016d28f21b5eebd608c34f13c8c35c253b8e0a55 |
| scipy_citation | https://raw.githubusercontent.com/scipy/scipy/main/CITATION.bib | retrieved | 00dae00c94b68345b59086ccff7183d7133f9b64e3045b0cced61daca7f67af4 |
| numpy_citation | https://raw.githubusercontent.com/numpy/numpy/main/CITATION.bib | retrieved | 6b12115280759df0c423d639828d2b95bb3837373f65136fcc5086cc5560d528 |
| sklearn_ridge | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/linear_model/_ridge.py | retrieved | f61f65518b672fe931cbb981aa4ecc620fc12309bc782e23615892c54121fc95 |
| sklearn_timesplit | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/model_selection/_split.py | retrieved | c8906abec0bb3af831928d4c3689c904ff59ed49e5dc99c4038f72d91ec2d093 |
| sklearn_common_pitfalls | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/common_pitfalls.rst | retrieved | 6e2b42374fbb521c35f29eeab05bcbfa9744eed344b5f92e41013338ff60a2f5 |
| sklearn_about | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/about.rst | retrieved | c792b39aa0074afe1c6eda11a1812d1df8bc03e33f04e11a24b076ca394bf769 |
| pytorch_randomness | https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/randomness.rst | failed | HTTP Error 404: Not Found |
| pytorch_benchmark | https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/benchmark/utils/timer.py | retrieved | cc8507321d5247054e7de987b5c0463d58aa26d0f7d083e7f698923464204813 |
| wavenet | https://raw.githubusercontent.com/ibab/tensorflow-wavenet/master/README.md | retrieved | 35d5570f555c3c873c6db96447a7d687cc6274f8b805d43f577d3ba278ad9d23 |
| film | https://raw.githubusercontent.com/ethanjperez/film/master/README.md | retrieved | 8d00f85d0fa03364bdf3fb8ef1dc72c87cccc03930785b4a05db6fee1cfdedb6 |
| sklearn_ar_example | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/applications/plot_time_series_lagged_features.py | retrieved | 49191414a48b76aa1ecea246d642ceb7e7a69dcda8197795ea0397c17707d04f |
| tscv | https://raw.githubusercontent.com/hudson-and-thames/mlfinlab/master/README.md | retrieved | 22bf05e1c98e4f0d518708c79d3316626e581a3c0dfb4e69ba1d4e3788ad36bb |
| tcn_model | https://raw.githubusercontent.com/locuslab/TCN/master/TCN/tcn.py | retrieved | f93c67de970f4c9e1a96a7d2bed1df89c521faf6ca94e36455267605b1ad585f |
| film_module | https://raw.githubusercontent.com/ethanjperez/film/master/vr/models/filmed_net.py | retrieved | 943bfed724c377763d467327d54416f57ffdcb8fb34827cd9cd5f51f74846bc9 |
| pytorch_randomness_v251 | https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/randomness.rst | retrieved | ed86e72eacbe7e6e8e667c0a32a30451936dd60eed3176747b87d14f4eb1291a |
| sklearn_citation | https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/CITATION.cff | retrieved | 3365069d76e8820628da2cc5de815bec16b44c0bd23ff7294111cd9c0ab2fb8a |
| matlab_envelope | https://www.mathworks.com/help/signal/ref/envelope.html | failed | <urlopen error Tunnel connection failed: 403 Forbidden> |
| matlab_kurtosis | https://www.mathworks.com/help/stats/kurtosis.html | failed | <urlopen error Tunnel connection failed: 403 Forbidden> |
| anchor_crossref | https://api.crossref.org/works/10.1016/j.ymssp.2024.112161 | failed | <urlopen error Tunnel connection failed: 403 Forbidden> |
| tcn_arxiv | https://arxiv.org/abs/1803.01271 | failed | <urlopen error Tunnel connection failed: 403 Forbidden> |
| gru_acl | https://aclanthology.org/D14-1179/ | failed | <urlopen error Tunnel connection failed: 403 Forbidden> |
| acl_gru_xml | https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/D14.xml | retrieved | 8e0776148cb82ba1052258d7644dc7b415fc7e2e3d1874de9ac63e9321818046 |
| adam_author | https://raw.githubusercontent.com/dpkingma/adam/master/README.md | failed | HTTP Error 404: Not Found |
| domain_UDTL | https://raw.githubusercontent.com/ZhaoZhibin/UDTL/master/README.md | retrieved | a5858ca9eb57a1452ee5d9092f11c52366e84fdc9b77481576cf077ef00e2c48 |
| domain_Weighted_Multi-Scale_Dictionary_Learning | https://raw.githubusercontent.com/ZhaoZhibin/Weighted_Multi-Scale_Dictionary_Learning/master/README.md | retrieved | d6c4af55d0d9f8af22d0ee749e914d7174b58134d3fade1400106a73720594ed |
| domain_Limited-Data-Rolling-Bearing-Fault-Diagnosis-with-Few-shot-Learning | https://raw.githubusercontent.com/SNBQT/Limited-Data-Rolling-Bearing-Fault-Diagnosis-with-Few-shot-Learning/master/README.md | retrieved | 66e52c1fc5dbec59e82cadda59e713fa12c7ac8b010c1a0054b56a4a70c79c46 |
| domain_Vibration-Based-Fault-Diagnosis-with-Low-Delay | https://raw.githubusercontent.com/Western-OC2-Lab/Vibration-Based-Fault-Diagnosis-with-Low-Delay/master/README.md | retrieved | b5936446d8508d6df39611e3baa128060e712449b76735511c3ac647abf62c19 |
| domain_A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method | https://raw.githubusercontent.com/zhaoyuzhi/A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method/master/README.md | retrieved | 6bef0c58b4a50535cfe0eedc3ddae0a6c8d11a5e994875a983994bfc87ef5e43 |
| domain_TFN | https://raw.githubusercontent.com/ChenQian0618/TFN/master/README.md | retrieved | 33caee3a2d0c0c182c90f56ff1bf7bd8ebf7003f54a42072760c0e247178f84b |
| domain_SQ-dataset-with-variable-speed-for-fault-diagnosis | https://raw.githubusercontent.com/shenliuuu/SQ-dataset-with-variable-speed-for-fault-diagnosis/master/README.md | retrieved | 38ece8f6c20ddd196c941faae9459e2b8f2734ef1c378c4e00f617eb03572c5e |
| domain_MCKD | https://raw.githubusercontent.com/xwy111/MCKD/master/README.md | retrieved | 2c0bd5d40180a7917de72bba91c52bde8b8ad593ea424885dc3246f937e5a0d9 |
| domain_MCKD | https://raw.githubusercontent.com/teataeteatae/MCKD/main/README.md | failed | HTTP Error 404: Not Found |
| domain_COT | https://raw.githubusercontent.com/xyxxyx1999/COT/main/README.md | failed | HTTP Error 404: Not Found |
| domain_rota-order | https://raw.githubusercontent.com/ZhenxianLi/rota-order/master/README.md | retrieved | b146ed2f0ce1c08e2f4cb0a578167454706e44a17fb62ec4f4abf0438aedccc8 |
| domain_Fast_SASD | https://raw.githubusercontent.com/ZhaoZhibin/Fast_SASD/master/README.md | retrieved | 6bf216c10608c50c9c38f5f793f0da4a87f74495315bb3a04ce1304ec9adb235 |
| domain_HHLP-for-weak-fault-feature-enhancement | https://raw.githubusercontent.com/ZhaoZhibin/HHLP-for-weak-fault-feature-enhancement/master/README.md | retrieved | f7a22421d257796a78f391b794c14c2e86673b83bac793c69a2047eb122d35fc |
| domain_SMPGL | https://raw.githubusercontent.com/ZhaoZhibin/SMPGL/master/README.md | retrieved | db84e0a3b087fcbc388da62a63eaa5bd0fbf1cad872b3291777fcab4834bafc5 |
| domain_AdaESPGL | https://raw.githubusercontent.com/ZhaoZhibin/AdaESPGL/master/README.md | retrieved | f3fecf0887ea6ea4685d8bf0c0fbf3491eecf041c36dd8d01aaa16485d870a41 |
| domain_WaveletKernelNet | https://raw.githubusercontent.com/HazeDT/WaveletKernelNet/master/README.md | retrieved | 8ea26d51d56613a98b90981e4c23dca8ae5b0cb9aff44248ae57170793b34183 |
| domain_Basic-Rotating-Machine-Vibration-Analysis | https://raw.githubusercontent.com/ZhaoZhibin/Basic-Rotating-Machine-Vibration-Analysis/master/README.md | retrieved | a0f09d569a64ea48dc8aabe07d640b9bb34fee994006fc0fc40ae4c4cbde2dfd |
| domain_DL-based-Intelligent-Diagnosis-Benchmark | https://raw.githubusercontent.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/master/README.md | retrieved | 8e822ca1ad8bcee5502cfc96da80715dbd616a77b93df3bc476623780aa95db8 |
| zhao_publications | https://raw.githubusercontent.com/ZhaoZhibin/ZhaoZhibin.github.io/master/_pages/publications.md | retrieved | 19dce237e9866c7dc6781fcf86cefdaadad71860612c8f8c70485e55e204d60d |
| zhao_codes | https://raw.githubusercontent.com/ZhaoZhibin/ZhaoZhibin.github.io/master/_pages/codes.md | retrieved | ce04656016debe97291c73bf40f400893239d382a7583da8b36e441cfaf4d9ad |

## Evidence depth and remaining gaps

The updated pool contains **19 publication records**: 12 additional machinery-diagnostics studies [R08–R19], three sequence/conditioning sources [R02–R04], three software-infrastructure citations [R05–R07], and the course-cited but inaccessible anchor [R01]. The 15 method-relevant author/abstract records exclude the three software citations and inaccessible anchor. Nine official methods documents [D01–D09] and one unverified candidate [U01] are separate. **No article full text was read.** Most machinery evidence is original author README method/abstract summaries; R19 is citation/code context only. This is a substantially enlarged, source-qualified review, not a claim of 19 full-text readings or completed anchor reproduction. Several related sparse-method sources share an author group, a selection bias noted in the synthesis.

The Chen et al. publisher metadata, full LPV-ARMA specification and directly comparable results remain inaccessible. Exact original MCKD/PMI and historical order-tracking paper methods were not recovered from an authoritative accessible source. These specific gaps are not repaired by the larger general diagnostics pool. The supplied MATLAB source and explicit synthetic numerical checks remain the implementation evidence for CK/PMI. The new split fault files are now locally accessible; unknown labels and speed units remain scientific limitations, not a file-transfer blocker.
