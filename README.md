# 3DFR — VGGT-Omega × AliceVision 人臉重建

| 檔案 | 用途 |
|---|---|
| `vggt_omega_to_alicevision.py` | 主管線：VGGT-Omega 推論 → AliceVision 輸入 → meshing / texturing |
| `texture_pyramid.py` | v4 貼圖：Laplacian／Gaussian 金字塔 + 光流對齊重新烘焙 AliceVision 圖集 |
| `av_mesh_io.py` | OBJ/MTL、SfMData 相機讀取、numpy 光柵化器、push-pull 補洞（兩者共用） |
| `VGGT_OMEGA_TO_ALICEVISION.md` | 說明與實驗紀錄；**§8 是 v4 的做法與 Multiface 驗證結果** |
| `viewer/` | 本機網格並排比較網頁（three.js，同一個滑鼠同時控制兩個模型） |
| `experiments/` | 在公開資料集 Multiface 上的評估腳本（不含任何資料，執行時下載） |

v4 建議指令見 `VGGT_OMEGA_TO_ALICEVISION.md` §8.5。

相依套件：`numpy`、`scipy`、`opencv-python`、`pillow`（新模組不需要 GPU／OpenGL），
以及原本管線需要的 `torch`、`OpenImageIO`、AliceVision 二進位檔。

## 網格並排比較（`viewer/`）

```bash
python viewer/serve.py --root C:/atop/data            # Windows
python viewer/serve.py --root /mnt/c/atop/data         # WSL
python viewer/serve.py --root C:/atop/data --root D:/other   # 多個資料夾
```

會自動開瀏覽器到 http://localhost:8800 （或在 Windows 雙擊 `viewer/start_viewer.bat`）。

* 左右各選一個 `.obj`（上方的篩選框可以輸入 `v4 texturedMesh_pyramid` 之類的關鍵字）；
  兩邊共用同一個鏡頭，拖曳旋轉、滾輪縮放、右鍵平移。
* **著色**：純貼圖顏色（不打光）／貼圖＋光照／無貼圖＋光照（看凹凸）／法線色；
  另可疊加線框、切換平面著色。快速鍵 `1` `2` `3` `4`、`W`、`F`。
* **光源**：跟著鏡頭或固定在模型；方位、仰角、強度、環境光可調。把仰角壓低成側光最容易看出凹凸。
* **版面**：左右／上下、交換（`S`）、上下翻轉、重設視角（`R`）、截圖（存成 PNG）。
* **共用座標**（預設）讓同一次拍攝的不同版本保持相同位置；不同拍攝互相比較時改用「各自置中」。
* 不在掃描資料夾裡的模型：把 `.obj` 連同 `.mtl` 和貼圖一起拖到左半或右半邊。

伺服器只在本機（127.0.0.1）監聽，three.js 已內含在 `viewer/vendor/`，不需要網路。
第一次開一個 100 萬面的網格，伺服器轉檔約 10–15 秒，之後同一個檔案會用快取。
