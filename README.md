# Take a Break in the Middle: Investigating Subgoals towards Hierarchical Script Generation
Goal-oriented Script Generation is a new task of generating a list of steps that can fulfill the given goal. In this paper, we propose to extend the task from the perspective of cognitive theory. Instead of a simple flat structure, the steps are typically organized hierarchically — Human often decompose a complex task into subgoals, where each subgoal can be further decomposed into steps. To establish the benchmark, we contribute a new dataset, propose several baseline methods, and set up evaluation metrics. Both automatic and human evaluation verify the high-quality of dataset, as well as the effectiveness of incorporating subgoals into hierarchical script generation. Furthermore, We also design and evaluate the model to discover subgoal, and find that it is a bit more difficult to decompose the goals than summarizing from segmented steps.

## Paper Link
[Download Paper](https://aclanthology.org/2023.findings-acl.644.pdf)

## Dataset
Instructable dataset is available at:
[Download Dataset](https://entuedu-my.sharepoint.com/:x:/g/personal/xinze_li_staff_main_ntu_edu_sg/ER2Xu6HUxzhEkR9jPjkbwh8BdZmYa9yQGQuHm9kcOUXZ8w)

## Citation
Please cite our paper if find our work helpful your work:
```bibtex
@inproceedings{DBLP:conf/acl/Li0CS23,
  author       = {Xinze Li and
                  Yixin Cao and
                  Muhao Chen and
                  Aixin Sun},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Take a Break in the Middle: Investigating Subgoals towards Hierarchical
                  Script Generation},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {10129--10147},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.644},
  doi          = {10.18653/V1/2023.FINDINGS-ACL.644},
  timestamp    = {Wed, 10 Apr 2024 15:48:06 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/Li0CS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
