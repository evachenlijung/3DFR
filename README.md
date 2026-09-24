# 3DFR — VGGT-Omega × AliceVision 人臉重建

| 檔案 | 用途 |
|---|---|
| `vggt_omega_to_alicevision.py` | 主管線：VGGT-Omega 推論 → AliceVision 輸入 → meshing / texturing |
| `texture_pyramid.py` | v4 貼圖：Laplacian／Gaussian 金字塔 + 光流對齊重新烘焙 AliceVision 圖集 |
| `av_mesh_io.py` | OBJ/MTL、SfMData 相機讀取、numpy 光柵化器、push-pull 補洞（兩者共用） |
| `VGGT_OMEGA_TO_ALICEVISION.md` | 說明與實驗紀錄；**§8 是 v4 的做法與 Multiface 驗證結果** |
| `experiments/` | 在公開資料集 Multiface 上的評估腳本（不含任何資料，執行時下載） |

v4 建議指令見 `VGGT_OMEGA_TO_ALICEVISION.md` §8.5。

相依套件：`numpy`、`scipy`、`opencv-python`、`pillow`（新模組不需要 GPU／OpenGL），
以及原本管線需要的 `torch`、`OpenImageIO`、AliceVision 二進位檔。
