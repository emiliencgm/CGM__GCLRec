# CGM__GCLRec
## TODO
1. 检查梯度的流动，尤其是对比损失中增强视图的梯度流动方式是否正确。
2. 调整loss和augment中自适应系数的计算方法。
3. Adaptive_Neighbor_Augment中的邻域信息扰动方法
4. 两个对比视图的选择（origin vs augment， augment vs augment等）
5. baselines的效果检验
6. 优化部分代码，提高执行效率
7. 实现‘mlp’自适应，类比BC loss，将多种先验知识（或其embedding）输入联合训练的mlp，得到综合自适应系数。
8. 设计实验（推荐效果对比、分组对比、可视化分析、消融实验、训练效率、抗噪声）
9. 增加更多常见baseline，如MF
10. 考虑除homophily外，无连接的节点之间的重要性度量方法
11. tensorboard的展示方法
12. 优化cpu和cuda的计算量分配; 调整计算adaptive_coef等时不同数据（包括batch_user等索引）所处的设备
13. commonNeighbor的torch_sparse稀疏张量的快速批量索引
14. homophily的KMeans加速（莫名其妙地自己加速了……），以及考察使用全部/部分embedding进行聚类的效果的差异
15. user/item分开聚类？
## 使用方法
可使用nohup指令在服务器后台运行，如：`nohup python -u runs.py --cuda 0 > nohups/runs.out 2>&1 & `，可改成`>> nohups/runs.out`变为追加写





# Remote Server SSH Config
```
Host 外网新主2080x2(姜)
    HostName 657yz61365.zicp.fun
    User cgm
    Port 17296

Host 外网新主3090x1(壮)
    HostName 657yz61365.zicp.fun
    User cgm
    Port 29475

Host 外网114-1
    HostName 3s6252y571.zicp.vip
    User cgm
    Port 31401

Host A716
    HostName 10.134.12.239
    User a716

Host 内网新主2080x2(姜)
    HostName 10.134.148.148
    User cgm
    port 22

Host 内网新主3090x1(壮)
    HostName 10.134.148.185
    User cgm
    port 22

Host 内网114-1
    HostName 10.130.104.15
    User cgm
    Port 9125
```