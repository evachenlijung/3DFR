# VGGT-Omega → AliceVision 紋理化網格重建管線

`vggt_omega_to_alicevision.py` 把 VGGT-Omega 的推論結果（相機姿態 + 稠密深度）轉成
AliceVision 原生輸入格式，然後直接呼叫 AliceVision 的 meshing / meshFiltering /
texturing 產出帶貼圖的 `.obj`。

> **最後更新 2026-09-24。** §1–§5 是管線本身的說明；**§7 是實驗紀錄**——試過什麼、
> 結果如何、下一步測什麼，以及哪些舊結論已經被推翻。先看 §7 再看前面，
> 因為 §2.5 和 §5.5 有幾段已作廢的建議（就地標註了）。
> **§8 是 v4**：貼圖改用 Laplacian／Gaussian 金字塔重新烘焙、幾何修正像素中心偏移，
> 在公開資料集 Multiface 上以與拍攝設備相同的 7+2 相機配置驗證。

```
images/ ──[VGGT-Omega]──► poses + dense depth
        ──[本腳本]───────► sfm.sfm + depthMaps/*.exr + undistorted/*.png
        ──[aliceVision_meshing]───────► Delaunay 四面體化 + graph-cut 網格
        ──[aliceVision_meshFiltering]─► 平滑 / 保留最大連通元件
        ──[aliceVision_meshDecimate]──► （選用）簡化
        ──[aliceVision_texturing]─────► texturedMesh.obj + .mtl + texture_*.png
```

---

## 1. 快速開始（Ubuntu）

### 1.1 建立 conda 環境

```bash
conda create -n vggt_omega_alicevision python=3.10 -y
conda activate vggt_omega_alicevision

# PyTorch（依你的 CUDA 版本調整；範例為 CUDA 12.1）
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y

# 本專案 + EXR 讀寫
pip install -r requirements.txt && pip install -e .
conda install -c conda-forge openimageio -y   # 較佳；沒裝的話腳本會退回 opencv-python 寫 EXR（缺 metadata但仍可跑）
```

沒有 conda-forge 的 OpenImageIO 也可以，退而求其次：

```bash
pip install opencv-python
```

### 1.2 執行

```bash
python vggt_omega_to_alicevision.py \
    --images     /data/albert_185119/images \
    --masks      /data/albert_185119/bgrm/dilation/masks \
    --checkpoint ./checkpoints/vggt_omega_1b_512.pt \
    --av-bin     /opt/Meshroom-2023.3.0/aliceVision/bin
```

不指定 `--output` 時，輸出資料夾會自動建立在 `--images` 的**同一層目錄**，
並以 images 資料夾名稱加上 `_alicevision` 當前綴，方便直接對照是哪組輸入產生的：
`/data/albert_185119/images` → `/data/albert_185119/images_alicevision`。
需要輸出到別的地方時仍可用 `--output <path>` 覆寫。

批次處理（自動偵測每個 capture 的 `bgrm/*/masks` 與 `brightness-CCT-adjust/*`，
每個 capture 一樣會各自產生自己的 `images_alicevision/`）：

```bash
python vggt_omega_to_alicevision.py \
    --batch-root /data/20250415/0415 \
    --checkpoint ./checkpoints/vggt_omega_1b_512.pt \
    --av-bin     /opt/Meshroom-2023.3.0/aliceVision/bin \
    --continue-on-error
```

輸出結構：

```
<images 同層目錄>/<images資料夾名>_alicevision/
├── sfm.sfm                     AliceVision SfMData（views / intrinsics / poses）
├── depthMaps/                  <viewId>_depthMap.exr, <viewId>_simMap.exr
├── undistorted/                <viewId>.png  ← texturing 的來源影像
├── mesh/                       densePointCloud.abc, rawMesh.obj, filteredMesh.obj
└── texturedMesh/               texturedMesh.obj + .mtl + texture_1001.png
```

例如 `--images /data/albert_185119/images` 會產生：

```
/data/albert_185119/
├── images/
├── bgrm/
└── images_alicevision/
    ├── sfm.sfm
    ├── depthMaps/
    ├── undistorted/
    ├── mesh/
    └── texturedMesh/
```

---

## 2. 皮膚色不連續怎麼解

這是這個資料集最實際的問題，而且**通常不是 texturing 的 bug，是各視角的曝光 / 白平衡漂移**。

我在你的兩組真實資料上量測了每個視角「臉部遮罩區域的中位膚色」：

| capture | 來源 | L\* 標準差 | L\* 全距 | b\* 全距 |
|---|---|---|---|---|
| albert_185119 (57 views) | `images/` 原圖 | 3.47 | **24.00** | 16.17 |
| albert_185119 | `brightness-CCT-adjust/` | 2.26 | 16.30 | 14.95 |
| albert_185119 | `--harmonize gain` | **0.22** | **0.84** | 8.02 |
| tiffany_183741 (60 views) | `images/` 原圖 | 3.99 | **33.82** | 16.44 |
| tiffany_183741 | `--harmonize gain` | **0.23** | **0.91** | 6.29 |

L\* 差 24–34 單位是肉眼非常明顯的差異；貼圖時不同三角形取到不同視角，就會出現色塊。
`docs/skin_swatches.png` 把每個視角的中位膚色畫成色條，可以直接看到原圖與
`brightness-CCT-adjust` 都還有明顯的條紋，而 `--harmonize gain` 之後是均勻的一片。

腳本的處理方式（預設開啟）：

* **`--harmonize gain`**（預設）：在**線性 RGB**空間對每個視角套一個 per-channel
  von Kries 增益，把該視角遮罩區域的中位色對齊「所有視角的中位數」。
  只改低頻的曝光/白平衡，不動細節。
* `--harmonize luminance`：只做亮度增益（保留原本的色溫差異）。
* `--harmonize none`：關閉。若你的打光本來就刻意不同（例如要保留 shading），用這個。

搭配的第二道防線是 AliceVision 的 multi-band blending：

* `--multi-band-nb-contrib 1 5 10 0`（預設，與你現有 Meshroom 設定一致）。
  最後幾個 band 是低頻，數字越大代表低頻色彩由越多視角平均 → 殘留色差越不明顯。
  若還看得到色塊，先把它調成 `1 5 20 10`。
* `--use-score` / `--best-score-threshold 0.1` / `--angle-hard-threshold 90`：
  只讓觀測角度好的視角貢獻顏色。
* `--masks`：把背景深度丟掉，避免背景幾何把背景色帶到臉部輪廓上。

**建議順序**：先用預設跑；若仍有色塊 →
提高 `--multi-band-nb-contrib` 低頻 band → 再考慮 `--conf-percentile 30` 讓幾何更乾淨。

---

## 2.5 表面凹凸 / 龜裂紋路怎麼解（`--consensus-passes`）

重建出來的臉如果佈滿像乾掉泥巴那樣分岔的細溝，**原因不是 meshing 參數調錯，也不是
VGGT 的單張深度圖有問題**。

### 量測證據

拿 `patient_27_20260511_143743`（9 視角）把每個視角反投影到其他視角、和對方自己的深度
相比，單位是 **pixSize**（= depth / 焦距，也就是 AliceVision 所有融合容差使用的單位）：

| 送進 `aliceVision_meshing` 的深度圖 | 視角間誤差中位數 | 落在 2·pixSize 內 |
|---|---|---|
| VGGT-Omega 原始, res512 (k=7) | **3.51** | 33.8% |
| VGGT-Omega 原始, res1024 (k=3) | **5.14** | 23.9% |
| Meshroom SGM 原始 | 22.20 | 32.7% |
| **Meshroom 經過 DepthMapFilter**（目標） | **0.36** | **76.1%** |

同時，單張深度圖本身的粗糙度（|laplacian|，pixSize 單位）：
VGGT 0.24 vs Meshroom 濾波後 0.15 — **單視角其實很乾淨**
（`docs/depth_normals_compare.png` 的法線圖可以直接看出來，VGGT 的臉是平滑的）。

### 結論

Meshroom 的原始 SGM 深度圖其實比 VGGT 還亂（22.20），但它的 **DepthMapFilter** 節點會把
「少於 `minNumOfConsistentCams` 個視角佐證」的像素整片丟掉（84% → 29% 存活），所以
mesher 永遠只看到彼此吻合到 sub-pixel 的資料。

我們的管線把 VGGT 深度圖**直接寫進 filtered 槽**，完全跳過這一步。於是 graph-cut 拿到
9 層互相錯開 3.5 pixSize 的曲面，它就忠實地在層與層之間刻出縫隙 — 那就是你看到的龜裂紋路。

`template_decimation.mg` 裡 `Meshing` 節點幾乎全是 AliceVision 預設值，所以**答案不在
Meshing 的超參數**。你那張圖的成功之處是這條鏈多了三個節點：

```
DepthMap → DepthMapFilter → Meshing → MeshFiltering → MeshDecimate(0.2) → MeshDenoising → Texturing
                ^^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                我們原本缺的                             我們原本缺的
```

### 修正

新增 `cross_view_consensus()`，在寫出 EXR 之前跑。它做 DepthMapFilter 做的事，再多一步：
不只刪掉不一致的像素，而是把每個像素換成**有佐證的鄰近視角的中位數**，主動把幾層曲面拉在一起。
預設開啟（`--consensus-passes 3`）。

實測（用你既有的 `vggt_omega_pointmap_res512` 輸出跑，非模擬）：

| | 前 | 後 | Meshroom 目標 |
|---|---|---|---|
| 視角間誤差中位數 | 3.51 | **0.50** | 0.36 |
| 落在 2·pixSize 內 | 33.8% | **74.2%** | 76.1% |
| 單視角 \|laplacian\| | 0.24 | **0.06** | 0.15 |
| 有效像素 | 41.4% | 33.4% | 29.1% |

`docs/consensus_before_after.png` 是修正前後的法線圖。

相關參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--consensus-passes` | 3 | 0 = 關閉（回到會產生龜裂的行為） |
| `--consensus-tolerance` | 20.0 | 幾 pixSize 內算「佐證」；丟太多像素就調大 |
| `--consensus-min-agree` | **2**（舊 3） | 等同 `minNumOfConsistentCams`。3 會在下巴底／耳朵挖洞，見 §7.3(b) |
| `--consensus-neighbours` | 10 | 等同 `nNearestCams` |
| `--consensus-smooth-radius` | 2 | 共識後的邊緣保留平滑；嫌太糊就設 1 或 0 |

### 還有兩件事值得做

>  ⚠️ **2026-09-24 更新：下面第 1 點已被推翻。** 當時的比較是在「深度圖留在網路解析度」
>  的前提下做的，那個前提本身才是問題（見 §7.2）。現在 `--depth-resolution image` 是預設，
>  **正確做法是 `--image-resolution 1024`**，不是 512。

1. ~~**解析度不是越高越好**。res1024 的視角間誤差（5.14）比 res512（3.51）還差 —
   pixSize 縮小的速度比深度誤差下降的速度快。先用 `--image-resolution 512`。~~
   **（已作廢）** 視角間誤差用 pixSize 當單位，而 pixSize 本身隨深度圖解析度改變，
   所以這個比較是在拿兩把不同長度的尺互相比。改用固定的物理單位（mm）之後，
   res1024 優於 res512。
2. **補上後處理**。你的 `.mg` 有 `MeshDecimate(0.2)` + `MeshDenoising`。
   `--denoise-iterations 1` 現在是預設；`--decimate-factor` 仍是 0（不抽稀）。
   完整的 Meshroom 後處理鏈套上去會怎樣，量測見 §7.3(c)——打光起伏砍半，
   但頂點從 530k 掉到 106k，所以預設不開。

>  ⚠️ **2026-09-24 更新：下面這組「放大容差」的建議已作廢。** 它是在深度圖仍停留在網路
>  解析度時寫的，當時 pixSize 本身就已經被灌水 3.4 倍，再乘 4～8 倍等於把整張臉融成一團。
>  實測把 `--pix-size-margin-*` 往**小**調才是增密的方向（§7.3 的 M1–M6）。
>  正確的修法是修 pixSize 的來源（§7.2），不是補償它。

~~如果你想走「純調參」路線（不改深度圖），要把 mesher 的容差放大到符合資料的
3.5 pixSize 誤差，大約是 4 倍：~~

```bash
--consensus-passes 0 \
--pix-size-margin-init-coef 8 --pix-size-margin-final-coef 16 \
--n-pixel-size-behind 16 --sim-gaussian-size-init 30 --sim-gaussian-size 30 \
--min-vis 3 --min-step 4
```

這會讓表面變平滑，但同時也會犧牲真實的細節（容差放大是全域的）。
建議優先用 consensus，調參當作備援。

---

## 3. 貼圖解析度與 k

VGGT-Omega 的推論解析度低於原圖。AliceVision 要求**視角影像必須是深度圖的整數倍 k**
（`MultiViewParams` 用 `view.width / depthmap.width` 的整數除法推 k，然後假設
`getWidth() == depthmap.width`），k 由 `choose_image_scale()` 自動選。

### 3.1 `--image-resolution 1024` 實際跑出什麼尺寸

`image_resolution` **不是邊長，是 token 預算**（`_target_shape()` 的 `"balanced"` 模式）：

```
原圖 3000 x 4000，aspect = 4000/3000 = 1.33333   （在 [0.5, 2.0] 內，不裁切）
token_number = (1024 // 16) ** 2 = 64**2 = 4096          patch_size = 16
w_patches = sqrt(4096 / 1.33333) = 55.4256  -> round 55  -> 55*16 =  880
h_patches = 4096 / 55.4256       = 73.9008  -> round 74  -> 74*16 = 1184
                                                    網路輸入 / 深度圖 = 880 x 1184
```

`sqrt(880 * 1184) = 1020.7 ≈ 1024` — 這就是 `"balanced"` 的字面意思。
實際 token 數 `55 * 74 = 4070`（比預算少 0.6%，取整造成）。
縮放是非等向的（垂直多拉 0.91%），由 sfm 的 `pixelRatio = fy/fx` 吸收。

### 3.2 k 與匯出尺寸

```
ratio = 3000 / 880 = 3.409  ->  k = round(3.409) = 3
匯出影像 / 上採樣後的深度圖 = 2640 x 3552        （原圖的 88%）
```

注意 `choose_image_scale()` 用 **round 不是 floor**。3.409 落在 3.5 以下取 3；
如果原圖是 3200 寬，ratio 3.636 → k=4 → 匯出 3520 寬，**比原圖大**，等於放大照片。

**已知缺口**：純 Meshroom 的 texturing 讀 3000×4000，我們讀 2640×3552，
少 12% 線性解析度 = 23% 像素。待測的修法見 §7.8。

### 3.3 相關參數

* `--depth-resolution image`（預設）把深度圖重取樣到相機自己的像素網格，見 §7.2。
  `network` 是舊行為，只用來重現舊結果。
* `--image-scale N` 手動指定 k（0 = 自動）。
* `--max-image-side 8192` 限制上限。

---

## 4. 為什麼這樣接得上（實作細節）

以下三個慣例只要錯一個，網格就會整個歪掉。全部都對照
`C:\atop\AliceVision` 原始碼確認過，並用真實的 Meshroom cache
（`data/.../meshroom_output/DepthMap/*_depthMap.exr`）驗證過欄位與型別。

### 4.1 深度定義：沿射線距離，不是 z-depth

`MultiViewParams::backproject` 是

```cpp
P = CArr[cam] + (iCamArr[cam] * pix).normalize() * depth;
```

`.normalize()` 代表 AliceVision 的 depth 是**相機中心到該點的歐氏距離**，
而 VGGT-Omega 輸出的是 z-depth。轉換：

```
d = z * || ((x - cx) / fx, (y - cy) / fy, 1) ||
```

兩邊的像素索引慣例相同（整數索引，不加 0.5）。數值驗證：反投影結果與 VGGT 的
world point 誤差 4.9e-15。

### 4.2 SfMData JSON

* `poses[i].pose.transform.rotation` 是 **column-major**
  （`jsonIO::saveMatrix` 用線性索引，而 Eigen `Matrix3d` 預設 column-major）。
  rotation = 相機自世界的旋轉 R_cw，`center` = 世界座標下的相機中心 `-Rᵀt`。
* 內參存的是**毫米焦距 + pixelRatio + 主點相對影像中心的偏移**，不是 fx/fy/cx/cy。
* **版本相容性很重要**：`loadJSON` 會直接拒絕比自己新的檔案
  （"File has a version more recent than this library"）。你現有的 Meshroom 2023.3.0
  是 sfmDataIO **1.2.6**，而 `C:\atop\AliceVision` 原始碼已是 **1.2.14**。
  預設 `--sfm-version 1.2.6`（兩邊都吃得下）；1.2.6 走的是
  `setFocalLength(f_mm, ratio, useCompatibility=true)` 分支，
  所以 `fx = f_mm·W/sensorWidth, fy = fx·pixelRatio`。
  若確定跑新版 build，可用 `--sfm-version 1.2.14`（`fy` 與 `fx` 的角色對調）。

### 4.3 深度圖 EXR

檔名 `<viewId>_depthMap.exr` / `<viewId>_simMap.exr`，單通道 `Y`、float32，
放在同一個資料夾並用 `--depthMapsFolder` 傳給 meshing
（meshing 內部把它當成 `depthMapsFilterFolder`，讀的是 `depthMapFiltered` / `simMapFiltered`）。

有 OpenImageIO 時會寫入完整 metadata，型別與 AliceVision 自己寫的完全一致：

```
AliceVision:downscale   int     k
AliceVision:P           m44d    row-major 4×4，scale-1（= 影像解析度）的 K[R|t]
AliceVision:CArr        v3d     相機中心
AliceVision:iCamArr     m33d    Rᵀ·K_depth⁻¹（深度圖解析度）
AliceVision:nbDepthValues / minDepth / maxDepth
AliceVision:roiBegin*/roiEnd*/tileBuffer*/tilePadding
```

沒有 OpenImageIO 而改用 OpenCV 時 metadata 會缺，**但仍然可以跑**：
AliceVision 會退回用 `view.width / depthmap.width` 推 k、用 SfMData 重建投影矩陣。
這也是為什麼「影像寬必須是深度圖寬的整數倍」不能妥協。

simMap 值域 `[-1, 0]`，−1 最好（AliceVision 的融合權重是
`1 + (1 + sim) · simFactor`）。腳本把 VGGT 的 confidence 以 10–90 百分位正規化後映射進去。

---

## 5. 常用參數

> 預設值在 2026-09-24 改過一輪，理由與量測見 §7.3。舊預設寫在括號裡。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--image-resolution` | 512（**實務用 1024**） | VGGT token 預算，不是邊長；見 §3.1 |
| `--depth-resolution` | `image` | 深度圖重取樣到相機像素網格；`network` = 舊行為，會灌水 pixSize |
| `--conf-percentile` | **0**（舊 20） | 丟掉每個視角信心最低的 N% 像素。調高會在下巴／鼻翼挖洞 |
| `--depth-edge-rtol` | **0**（舊 0.03） | 丟掉 3×3 鄰域相對深度跳變過大的像素。單獨調整幾乎無效 |
| `--consensus-min-agree` | **2**（舊 3） | 等同 `minNumOfConsistentCams`。3 會砍掉只有 1–2 視角看得到的下巴底／耳朵 |
| `--denoise-iterations` | **1**（舊 0） | meshDenoising 次數 |
| `--multi-band-nb-contrib` | **1 1 10 0**（舊 1 5 10 0） | 頻段視角**增量**，不是每層的視角數；見 §7.5 |
| `--masks` | — | 前景遮罩（白＝主體），背景深度直接作廢 |
| `--texture-images` | — | 只給 texturing 用的另一份對齊影像（例如已校色版本） |
| `--min-step` | 2 | meshing 融合時的深度圖取樣步長，調 1 更密但更慢 |
| `--max-points` | 5,000,000 | 稠密點雲上限 |
| `--decimate-factor` | 0 | >0 才跑 meshDecimate |
| `--texture-side` | 8192 | 貼圖圖集邊長 |
| `--export-only` | — | 只產生 AliceVision 輸入，不跑二進位檔 |
| `--dry-run` | — | 只印出 AliceVision 指令 |
| `--output` | `<images>_alicevision` | 輸出資料夾；預設建立在 `--images` 同層目錄，見上方「1.2 執行」 |
| `--skip-inference` | — | 重用輸出資料夾裡既有的 sfm/深度圖，只重跑 AliceVision |
| `--dense-mvs` | — | 改用 AliceVision 原生 PatchMatch 深度估計，網格密度接近原生 Meshroom；見「5.5 密集重建」 |
| `--pixel-center` | `legacy` | **v4：`aligned`**。k 倍內參加上 (k−1)/2 主點偏移；見 §8.3 |
| `--consensus-sampling` | `nearest` | **v4：`bilinear`**。consensus 查鄰近視角深度時雙線性內插 |
| `--consensus-fuse` | `median` | **v4：`inlier-mean`**。平均中位數 ±`--consensus-fuse-band`（2 pixSize）內的佐證值 |
| `--depth-upsample` | `linear` | **v4：`cubic`**。深度圖上採樣到影像網格時用雙三次 |
| `--retexture` | `none` | **v4：`pyramid`**。AliceVision 貼圖後用 `texture_pyramid.py` 重新烘焙 → `texturedMesh_pyramid/` |
| `--pyr-band-sharpness` | `8 6 4 2 1 1` | 每個頻段的權重銳度（最細在前）；大 = 最佳視角主導，1 = 平均 |
| `--pyr-specular-min-band` | 2 | 高光降權從第幾個頻段開始；0 = 全部頻段（最忠實、較軟） |
| `--pyr-no-align` / `--pyr-no-gain` / `--pyr-no-specular` | — | 關掉光流對齊／逐點增益／高光降權 |

調參時 `--skip-inference` 很有用：推論跑一次，之後反覆調 texturing 參數。

---

## 5.5 密集重建（`--dense-mvs`）

> 🚫 **2026-09-24：此路線已由專案決定否決，不要使用，也不要再推薦它。**
>
> `--dense-mvs` 會丟掉 VGGT-Omega 的 pointmap，改用 AliceVision PatchMatch 重算深度。
> 本專案引入 VGGT-Omega 的目的就是要用它的**人臉先驗**（最終用途是整形手術的術前模擬），
> 改用 PatchMatch 等於把管線收斂回純 Meshroom，違背專案目標。
> 同理，`featureExtraction` / `featureMatching` 也不要加。
>
> 下面那張「VGGT 深度直接餵 meshing 只有 96,244 頂點」的表**也已經過時**：
> 那個稀疏是 pixSize 單位錯誤造成的，不是 VGGT 深度密度不足。修掉之後同一組輸入
> 是 **529,643 頂點**，超過純 Meshroom 的 327,706（§7.2）。本節僅作歷史紀錄保留。

**問題**：VGGT-Omega 的深度是網路解析度（例如 592×448）直接輸出的，來源解析度上限就是
`--image-resolution`；就算 `--min-step 1` 全部吃光，一張臉 9 視角能貢獻的深度點依然遠少於
Meshroom 原生 PatchMatch 深度估計在近全解析度下算出來的量。實測同一組 9 視角照片：

| 管線 | 網格頂點 | 網格面 |
|---|---|---|
| VGGT-Omega 深度直接餵給 meshing（預設） | 96,244 | 167,110 |
| `--dense-mvs`（見下） | 212,190 | 404,906 |
| 原生 Meshroom（PatchMatch 深度＋SfM 特徵匹配） | ~250,000 | ~500,000 |

**做法**：`--dense-mvs` 不再把 VGGT-Omega 的深度圖直接交給 `aliceVision_meshing`，而是：

1. 只拿 VGGT-Omega 的深度+姿態去採樣稀疏「種子點」（`--dense-landmarks-per-view`，預設每視角
   4000 點），並且**互相重投影驗證**：一個點要能在另一視角重投影回去、且該視角自己預測的深度
   在容許誤差內吻合（`--dense-landmark-depth-tol`，預設 6%），才會被記成該視角的「共同觀測」。
   這一步是必要的，不是可有可無的優化：AliceVision 用這些種子點做兩件事——
   * `aliceVision_depthMapEstimation` 選相鄰視角（`findNearestCamsFromLandmarks`）時，要求兩個
     視角之間有 20 個以上共同觀測、且夾角落在 `--dense-min-view-angle`～`--dense-max-view-angle`
     之間，否則兩視角互相都不算「鄰居」，深度估計直接跳過該視角（親測：只給單一觀測的點，
     `Found only 0/10 nearest cameras`，深度圖全部算不出來）。
   * `SgmDepthList::getMinMaxMidNbDepthFromSfM` 用種子點決定每個視角自己的深度搜尋範圍。
2. 把種子點寫進 `sfm.sfm` 的 `structure` 段（AliceVision 原生 SfMData landmark 格式），
   當作一般 SfM 三角化點使用（只是來源是 VGGT-Omega，不是特徵匹配）。
3. 用這份 sfm.sfm 對**全解析度**貼圖影像跑 `aliceVision_depthMapEstimation`（PatchMatch
   立體匹配，Meshroom 自己也是用這個）+ `aliceVision_depthMapFiltering`，取代 VGGT-Omega
   自己的深度圖，交給 `aliceVision_meshing`。

也就是說：VGGT-Omega 負責解決少視角/弱紋理下最難的相機姿態估計（傳統 SfM 特徵匹配在人臉這種
弱紋理、視角稀疏的場景常常失敗或不穩），AliceVision 自己的 PatchMatch 立體匹配負責把密度做到
接近原生 Meshroom 的水準。

**代價**：多了 `aliceVision_depthMapEstimation`（本例 67s）+ `aliceVision_depthMapFiltering`
（本例 18s）兩步，且需要 GPU（`aliceVision_depthMapEstimation` 是 CUDA-only，見下方 WSL 疑難
排解）。

**輸出位置**：`--dense-mvs` 時預設輸出資料夾改成 `--images` 同層的 `vggt_omega_dense_textured_mesh`
（而不是 `<images>_alicevision`），其餘子資料夾結構不變，最終貼圖網格在
`vggt_omega_dense_textured_mesh/texturedMesh/texturedMesh.obj`。

**已知限制**：這批測試資料沒有前景遮罩，背景/衣物等雜亂區域的深度本來就不穩定，PatchMatch
在這些區域算出來的深度比 VGGT-Omega 自己的深度更「碎」，貼圖上會看到明顯的雜訊色塊。人臉本體
的幾何與貼圖是乾淨對齊的；要澈底解決背景雜訊，需要補上 `--masks`（見「1. 快速開始」）。

**WSL + CUDA 疑難排解**：如果 `aliceVision_depthMapEstimation` / `aliceVision_depthMapFiltering`
噴 `cudaGetDeviceCount failed: no CUDA-capable device is detected`，通常是系統另外裝過原生
Linux NVIDIA 驅動（例如 `apt install nvidia-utils-*`），使得 `/lib/x86_64-linux-gnu/libcuda.so.1`
蓋過了 WSL 自己該用的轉接層 `/usr/lib/wsl/lib/libcuda.so.1`。解法是執行前確保
`LD_LIBRARY_PATH` 把 `/usr/lib/wsl/lib` 排在最前面：

```bash
export ALICEVISION_ROOT=/opt/Meshroom-2023.3.0/aliceVision
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ALICEVISION_ROOT/lib
```

---

## 6. 已驗證 / 未驗證

**已在離線環境驗證（幾何 / 格式層）**

* 幾何慣例的數值驗證：P 分解、iCamArr、z-depth→射線距離反投影誤差 4.9e-15；
  內參 mm↔px 來回轉換誤差 0；rotation column-major 來回無誤。
  慣例為 `X_cam = diag(1,-1,-1) . R . (X - C)`，`u = fx*x/z + cx`，`v = fy*y/z + cy`，
  `fx = focalLength * W / sensorWidth`，`fy = fx * pixelRatio`，principalPoint 是相對中心的偏移。
* EXR 輸出格式（單通道 `Y`、float32、尺寸）與真實 AliceVision 深度圖逐欄位比對一致。
* 膚色連續性量測與 `docs/skin_swatches.png`。

**已在使用者機器上實跑驗證（2026-09）**

* 完整管線在 WSL + conda `vggt_omega_alicevision` + RTX 4070 Ti (12 GB) 跑通，
  5 組 test set 全部 rc=0（§7.4）。
* 與使用者自己的純 Meshroom 快取（`<capture>/.cache_sfm_fixed/`）逐節點參數比對。
* AliceVision 原始碼在 `C:/atop/AliceVision/src/`。§7.5 的頻段行為是**直接讀原始碼
  並對照實跑 log 的三角形計數**確認的，不是推測。

**已知硬限制**

* `--image-resolution` 1280 / 1536 在 12 GB VRAM、9 視角下 OOM。1024 是目前上限。
* checkpoint 是 512 訓練的，1024 靠 RoPE `normalize_coords="max"` 外插。
* 輸入影像組固定，不會為任何一邊增加照片（VGGT-Omega 與 Meshroom 用同一組做對照）。

---

## 7. 實驗紀錄（2026-09）

臨床用途：**整形手術的術前模擬**。目標是高解析度、高精度、貼近真人臉的**帶貼圖**網格。
實驗範圍限定在 `C:/atop/data/test_set.txt` 列出的 5 組 capture（不是全部 32 組）。

執行環境：WSL + conda env `vggt_omega_alicevision` + RTX 4070 Ti (12 GB)。
所有腳本開頭都必須 export 這兩個變數，否則 AliceVision 直接噴
`libaliceVision_cmdline.so.3: cannot open shared object file`：

```bash
export ALICEVISION_ROOT=/opt/Meshroom-2023.3.0/aliceVision
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ALICEVISION_ROOT/lib
```

### 7.1 版本演進

| 版本 | 內容 | 輸出資料夾 |
|---|---|---|
| v1 | 深度圖留在網路解析度，`--image-resolution 512`，舊預設 | `photos_alicevision/` |
| v2 | `upsample_to_image_grid()` + `--image-resolution 1024` | `photos_alicevision_v2/` |
| v3 | v2 + 放寬三個過濾器 + `--multi-band-nb-contrib 1 1 10 0` + `--denoise-iterations 1` | `photos_alicevision_v3/` |

v3 的執行指令：

```bash
python vggt_omega_to_alicevision.py \
  --images $D/photos --masks $D/masks_david --output $D/photos_alicevision_v3 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt --av-bin $ALICEVISION_ROOT/bin \
  --image-resolution 1024 --device cuda
```

### 7.2 根因：pixSize 的單位被深度圖解析度灌水

**這是整輪實驗最重要的發現。**

`pixSize = depth / focal`，而 focal 是**以深度圖像素為單位**的。AliceVision `Meshing`
所有融合容差（`pixSizeMarginInitCoef` / `pixSizeMarginFinalCoef` / `nPixelSizeBehind` …）
都乘在 pixSize 上。深度圖線性方向小 k 倍 → pixSize 在**真實世界尺度**上大 k 倍。

直接比對使用者自己的純 Meshroom 快取，兩邊的 meshing 參數**完全一樣**
（`pixSizeMargin 2/4`、`minStep 2`、`minVis 2`、`simGaussianSize 10`），結果卻差 7 倍：

| 送進 `aliceVision_meshing` 的深度圖 | 尺寸 | 有效取樣點 | 原始網格 |
|---|---|---|---|
| Meshroom DepthMapFilter (downscale 1) | 3000 × 4000 | 16,187,790 | 327,706 v |
| 我們，網路解析度 (k=3) | 880 × 1184 | 2,429,432 | **45,593 v** |

雙重懲罰：取樣點少 6.7 倍，**而且**每個取樣點的融合半徑在世界尺度上大 3.4 倍。

**修法**：`upsample_to_image_grid()`（在 `export_frames()` 之前）把深度圖重取樣回相機
自己的像素網格，同時把 sfm 寫的內參換成影像解析度的 `K_img`、`AliceVision:downscale`
改成 1。**這兩件事必須一起做**——AliceVision 是從「深度圖尺寸 + 你宣告的內參」算 pixSize
的，只放大陣列不改宣告等於沒改。

重取樣不是單純內插：深度不連續處（3×3 erode/dilate 差值 > `rtol`）與無效像素邊界用
`INTER_NEAREST`，其餘用 `INTER_LINEAR`。用 nearest 是為了不要把前後景之間的跳躍內插成
一層假的斜面。

> 踩過的坑：第一版是把這些 guard 像素**直接丟掉**，臉的覆蓋率掉到 92.1%（鼻翼整圈不見）。
> 改成 nearest 重取樣後，保留率回到 25.9%，與網路解析度下完全一致。

**上取樣不會增加任何新的幾何資訊。** 那 880×1184 個深度值就是 VGGT 知道的全部。
網格變好只有兩個原因：(1) pixSize 恢復正確，Meshing 不再過度合併；(2) 同一誤差攤在更多
頂點上，graph-cut 解變平滑。它**不會**讓 VGGT 看到更細的結構——這就是眼睛銳利度至今
仍輸 Meshroom 的原因（§7.6）。

### 7.3 消融實驗

**(a) Meshing 密度參數**（`scratchpad/exp_geom.sh`，patient_27_20260519_181536）

`amp` / `coh` 是多尺度同調性指標，`.004` / `.032` 是臉部 bbox 對角線的比例。

| cfg | pixSizeMargin init/final | decimate | 頂點 | 面積% | >10° | amp.004 | coh.004 | amp.032 | coh.032 |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 2.0 / 4.0 | 1 | 41,243 | 100.0% | 2.79% | 0.134 | 0.330 | 1.799 | 0.824 |
| M2 | 1.0 / 2.0 | 1 | 93,377 | 101.9% | 1.73% | 0.085 | 0.672 | 1.894 | 0.828 |
| M3 | 0.5 / 1.0 | 1 | 215,345 | 102.2% | 1.20% | 0.077 | 0.773 | 1.974 | 0.815 |
| M4 | 0.25 / 0.5 | 1 | 371,648 | 98.1% | 0.93% | 0.077 | 0.825 | 2.031 | 0.823 |
| M5 | 0.25 / 0.5 | 2 | 371,875 | 97.8% | 0.78% | 0.075 | 0.810 | 1.998 | 0.823 |
| M6 | 0.5 / 1.0 | 0 | 215,234 | 103.1% | 3.71% | 0.102 | 0.775 | 2.032 | 0.786 |

結論：把 margin 調**小**才會增密（與 §2.5 舊建議相反）。但這只是補償手段；
修掉 §7.2 的根因之後，預設的 2.0/4.0 就能拿到 53 萬頂點。
`densifyNbFront` / `densifyNbBack` / `densifyScale` 試過，是災難（3,470 頂點），不要碰。

**(b) 補洞**（`scratchpad/ablate_holes.sh`，patient_27_20260522_135730，baseline `--min-vis 2`）

| | 改動 | 保留像素 | 頂點 |
|---|---|---|---|
| V0 | baseline | 24.2% | 328,033 |
| V1 | `--consensus-min-agree 2` | 28.6% | 403,248 |
| V2 | `--conf-percentile 0` | 32.7% | 450,386 |
| V3 | `--depth-edge-rtol 0` | 24.2% | **329,308（單獨幾乎無效）** |
| V4 | 以上三者全開 | **37.1%** | **529,533** |
| V5 | V4 + `--min-vis 1` | 37.1% | 530,174 |

* 下巴的洞單靠 `--consensus-min-agree 3 → 2` 就補起來了。這三個過濾器砍掉的，正好是
  VGGT 人臉先驗最有價值的地方（下巴底、鼻翼、只有 1–2 視角看得到的耳朵）。
* 內部破洞數 1 → 0（`scratchpad/holes.py`，用 scipy connected components 數 boundary loop）。
* **`--min-vis 1` 沒有好處**（+641 頂點）而且會重新引入小洞。維持 2。
* 共識參數也掃過（`scratchpad/cons_exp.sh`）：`--consensus-tolerance 5` 會把有效像素
  從 38.1% 砍到 10.8%，太兇。維持 20.0。

**(c) 純 Meshroom 後處理鏈套到我們的網格上**（`scratchpad/mr_post.sh`）

```bash
aliceVision_meshDecimate  --simplificationFactor 0.2
aliceVision_meshDenoising --denoisingIterations 5 --lambda 2.0 --eta 1.8 --mu 1.5 --nu 0.3
aliceVision_texturing     --multiBandNbContrib 1 1 10 0
# -> decimated = 105,928   denoised = 105,928   textured = 280,007 tri
```

打光起伏 83.78 → **42.65**，貼圖細節不變（32.39 → 32.37）。
**如果要 Meshroom 那種「乾淨平滑」觀感，這是現成的槓桿**，代價是 530k → 106k 頂點。
預設不開，因為術前模擬要的是精度不是觀感。

### 7.4 v3 五組測試結果

```
[1/5] patient_27_20260522_153941 rc=0 v=658,955
[2/5] patient_27_20260522_135730 rc=0 v=529,643
[3/5] patient_27_20260522_115457 rc=0 v=662,363
[4/5] patient_27_20260520_165505 rc=0 v=632,261
[5/5] patient_27_20260519_181536 rc=0 v=607,666
```

輸出在各 capture 的 `photos_alicevision_v3/texturedMesh/texturedMesh.obj`。

**與純 Meshroom 的整體表面比較**（遮罩侵蝕 31px，全可見表面，與取景無關）：

| | 網格頂點 | 貼圖色調不均 | 貼圖高頻細節 | 幾何打光起伏 |
|---|---|---|---|---|
| 純 Meshroom (1/5/10) | 74,040 | 55.26 | 32.58 | 98.20 |
| 我們 v2 (1/5/10) | 329,473 | 39.95 | 23.45 | 68.08 |
| 我們 v3 (1/1/10) | **529,643** | 45.47 | **32.39** | 83.78 |
| v3 + MR 後處理 | 105,928 | 46.08 | 32.37 | 42.65 |

**「純 Meshroom 貼圖完美」這個印象，用整體表面量測並不成立**：高頻細節兩邊打平
（32.58 vs 32.39），色調不均反而是 Meshroom 高（55.26 vs 45.47）。
**保留的但書**：兩個網格覆蓋的表面範圍不同，侵蝕後的遮罩區其實不是同一塊解剖結構。

貼圖 texel 密度也不是解釋：臉上 texel 數 Meshroom 3,006 vs 我們 v3 2,892（差 4%）。

### 7.5 多頻段混合的真實行為（讀原始碼確認）

來源：`C:/atop/AliceVision/src/aliceVision/mesh/Texturing.cpp` 與
`.../image/imageAlgo.cpp` 的 `laplacianPyramid()`。

**三件容易搞錯的事：**

1. **0 會被刪掉。** `m.erase(std::remove(m.begin(), m.end(), 0), m.end())`，
   所以 `[1,1,10,0]` → `[1,1,10]`，`nbBand = 3`，**不是 4 個頻段**。
2. **那些數字是增量，不是每層的視角數。** 緊接著 `std::partial_sum(...)`，
   `[1,1,10]` → 累積 `[1,2,12]`。
3. **被指派到第 b 層的視角會同時貢獻到 b 以下所有層**
   （`for (bandContrib = band; bandContrib < pyramidL.size(); ++bandContrib)`），
   所以最佳視角進了全部三層。

`downscaleCoef = pow(multiBandDownscale, band)`，`multiBandDownscale = 4`：

| 頻段 | 影像解析度 | 特徵週期（影像 px @2640） | 臉上的物理尺度 |
|---|---|---|---|
| band 0 | 2640（全解析） | < ~8 | **< 1.07 mm** |
| band 1 | /4 = 660 | ~8 – 32 | **1.07 – 4.30 mm** |
| band 2 | /16 = 165 | > ~32 | **> 4.30 mm** |

（1 影像 px 在臉上 = 0.134 mm，由 fx = 2549 px @2640、z = 343 mm 推得。）

**實際平均視角數**，從 v3 的 log 逐相機加總三角形數算出來（不是推測）：

```
band 0: 1,214,220 三角形貢獻 = 網格三角形數   -> 1.00 個視角
band 1: 1,213,554                             -> 2.00 個視角
band 2: 4,360,305                             -> 5.59 個視角
```

**重要推論**：`nbContribMax = min(m.back(), 可見相機數)`。我們只有 9 張照片，
`1/5/10` 的 16 和 `1/1/10` 的 12 都超過 9，所以 **band 2 在 v2 和 v3 是一樣的 5.59 視角**。
v2 → v3 唯一實際改變的是 **band 1：約 5.6 視角 → 2 視角**。

也因為是 `partial_sum` 且不能有 0，**band 1 的累積值最低就是 2**，
三頻段設定下已經壓不下去了。

### 7.6 眼睛模糊：是幾何問題，不是解析度問題

多視角平均要成立，前提是網格準到 N 個視角投影下去會落在同一個物理點。
網格深度差 δ 時的重投影偏移：`Δu ≈ fx · B · δz / z²`。

實測參數：fx = 2549 px @2640、z = 343 mm、鄰近視角基線 B = 56 mm、一般視角對 B = 153 mm。

| 深度誤差 | 鄰近視角偏移 | 一般視角對偏移 |
|---|---|---|
| 0.2 mm | 0.24 px | 0.66 px |
| 0.5 mm | 0.61 px | 1.67 px |
| 1.0 mm | 1.22 px | 3.34 px |
| 2.0 mm | 2.44 px | 6.68 px |

對照尺度：

```
1 影像像素在臉上 = 0.134 mm
1 VGGT 網路像素  = 0.458 mm      <- 幾何的真實取樣間距
睫毛寬          = 0.1 - 0.3 mm  <- 比幾何取樣間距還小
```

**幾何取樣間距（0.458 mm）比貼圖取樣間距（0.134 mm）粗 3.4 倍**，也比睫毛本身粗。
眼瞼緣那個 2–3 px 的深度階梯在 880×1184 的格子上根本不存在。

對照 §7.5 的頻段尺度：

* 睫毛（0.1–0.3 mm）→ **band 0 → 1 個視角 → 不會被平均糊掉**，
  但 texel 透過錯誤的網格取樣，會被**拉伸扭曲**。
* 眼瞼摺痕 / 睫毛叢 / 虹膜邊緣（1–4 mm）→ **band 1 → 2 個視角 → 兩份錯開約 1–3 px 疊加**。
  v2 時這一層是約 5.6 份疊加，所以更糊。

改 `1/5/10 → 1/1/10` 讓眼睛銳利度 13.19 → 15.00（同一網格上量），
**影像解析度一個像素都沒動**——所以瓶頸不是解析度。

### 7.7 膚色不均（鼻尖、下唇）

在 440,674 個表面點上分解同一點的跨視角亮度差異：

```
總差異               26.5%
  每張照片的曝光項     9.4%   <- --harmonize gain 只能碰這一項
  空間項（陰影+鏡面） 24.7%   <- 真正的問題
```

實測 harmonize 前後：**26.50% → 26.47%，等於沒作用**。它估的是「每張影像一個標量增益」，
但每個視角看到的解剖結構不同，所以它量到的是構圖不是曝光。

原始照片上的鏡面高光（99 百分位亮度連通區域，等效直徑 @3000px）：

```
photo 0:  186, 69, 59, 47, 40 px
photo 4:   14, 10,  9,  8,  8 px   <- 這個視角幾乎沒高光
photo 8:  174, 157, 82, 34, 29 px
```

強烈視角相依。換算到 @2640 是 26–167 px → **落在 band 2（> 4.30 mm）**，
而 band 2 在 v2 和 v3 是一樣的（§7.5）。

所以 v2 → v3 的斑塊度退步（39.95 → 45.47）**不是高光核心被單視角烘進去**，
而是 **band 1（1.07–4.30 mm）從約 5.6 視角降到 2 視角**——高光的邊緣過渡、油光細紋、
皮膚微陰影在那個尺度失去了平均。

鼻尖和下唇是全臉曲率最高的位置，鏡面波瓣最窄、視角相依最強，
所以每個 test case 都在同樣位置出問題。

**這是一個旋鈕的兩端，band 參數本身解不開：**

```
band 1 多視角  ->  皮膚勻   + 眼睛被錯位塗抹
band 1 少視角  ->  眼睛銳利 + 中頻視角差異留下接縫
```

### 7.8 量測方法的已知問題（重要）

* **固定比例方框取樣不是取景無關的。** 眼睛銳利度與斑塊度的數字會隨 render 的 `pad`
  改變（pad 2.6 → 2.2 時，眼睛銳利度從「Meshroom 23.19 / 我們 15.96」翻成「18.61 / 19.05」）。
  **那組數字不能當結論。** 現在一律改用整個可見表面（遮罩侵蝕）取樣。
* **跨視角光度一致性指標是失敗的。** 把取樣點平移 20 px，相關係數從 0.003 變成 0.003，
  零訊號。不要再用它。
* **同調性指標分不出「同調的假影」和「同調的細節」。** 一道波紋和一道真實的皺褶對它
  是一樣的。必須配合肉眼看圖。
* 太細的尺度會漏掉大尺度起伏，所以評估一定要包含 r = 0.032 / 0.064。
* 粗尺度的鄰域矩陣在 37 萬頂點上會 OOM，需要按 `1/alpha²` 比例做頂點抽樣。

### 7.9 下一步待測

按優先序：

1. **`--multi-band-nb-contrib 1 0 0 0` 診斷**（成本：只重跑 texturing，約 4 分鐘）。
   0 被刪 → `[1]` → `nbBand=1` → 單頻段、全解析度、單一視角、**完全不混合**。
   * 眼睛**還是**不脆 → 剩下的全是幾何扭曲，光度精修是唯一的路。
   * 眼睛明顯變脆 → band 1 的雙重影才是主因，還有調整空間。

2. **`--image-resolution 1152`**（成本：重跑推論）。
   目的是讓 k=3 剛好落在原圖上，同時補掉 §3.2 那 12% 的缺口：
   ```
   (1152//16)^2 = 72^2 = 5184 tokens
   w_patches = sqrt(5184/1.33333) = 62.35 -> 62 -> net_w = 992
   h_patches = 5184/62.35         = 83.14 -> 83 -> net_h = 1328
   匯出 = 992*3 x 1328*3 = 2976 x 3984       （原圖的 99.2% / 99.6%）
   ```
   兩邊同時受益：texturing 拿到近乎全解析度的影像，VGGT 的深度格點也從 880 → 992（+12.7%）。
   **風險**：token 4,070 → 5,146（+26%），attention 成本約 1.6×，12 GB 可能 OOM，要實測。

3. **鏡面成分處理**（這才是膚色不均的正解，§7.7）。
   * 光學解：拍攝時加**交叉偏振片**。**這個問題還沒得到使用者答覆**——
     設備能不能上偏振片會決定要不要走計算解。
   * 計算解：texturing 前做 diffuse / specular 分離，只對 diffuse 貼圖。
     `--texture-images` 這個掛勾可以直接接上，不用改管線結構。

4. **光度精修**（VGGT 深度 vs 全解析度原圖）。最有價值但最花工。
   把對齊誤差從 ~1 mm 壓到 ~0.2 mm，投影偏移就從 3.34 px 降到 0.66 px，
   那時候就能切回 `1/5/10`，眼睛銳利度與膚色均勻度兩邊都拿到。
   這也是「為什麼引入 VGGT 還能更好」最強的論述。

5. **零碎的待試項**：`--consensus-smooth-radius 2 → 0`；貼圖頻段推到 `1/1/1`
   （注意：累積 `[1,2,3]`，低頻段會從 5.59 掉到 3 個視角，膚色連續性會變差）。

6. **全 32 組 capture 的 v3 批次**——**使用者明確要求先不要跑**，等 5 組測試確認完再說。

### 7.10 尚未處理的雜項

* 10 個 `photos_alicevision/texturedMesh/` 資料夾裡有 44 張 09-03 的舊 PNG（約 400 MB），
  已提議刪除，尚未得到答覆。

### 7.11 環境踩雷紀錄

* `wsl bash /mnt/c/...` 在 Git Bash 下會被路徑改寫 → 指令前面加 `MSYS_NO_PATHCONV=1`。
* Windows 端的 Python 讀不到 `/mnt/c/` → Python 用 `C:/...`，WSL 用 `/mnt/c/...`。
* Windows 終端 `cp950` 編碼會在 `‰` 之類的字元炸掉 → 設 `PYTHONIOENCODING=utf-8`。
* WSL 的 OpenCV 預設停用 EXR → 讀深度圖前先 `export OPENCV_IO_ENABLE_OPENEXR=1`。

---

## 8. v4：金字塔重新貼圖 + 幾何一致性修正（2026-09-24）

三個目標：(1) 用 Gaussian / Laplacian 金字塔保留皮膚高頻細節，特別是眼周；
(2) 貼圖無接縫；(3) 模型表面的凹凸再降低。

### 8.1 怎麼驗證的（公開資料集，不含任何病患資料）

私人資料不能放進公開 repo，所以改在 **Multiface**（Meta，CC BY-NC 4.0，受試者
6795937，表情 E001，第 102 格）上測，相機配置照拍攝設備挑：

| 角色 | Multiface 相機 | 方位角 / 仰角 |
|---|---|---|
| 7 台與眼睛同高 | 400053, 400042, 400012, 400016, 400013, 400028, 400059 | −71°…+76°，仰角 ±9° 內 |
| 2 台下巴下方往上拍 | 400031, 400061 | 仰角約 +45° |
| 保留評分（不參與貼圖） | 400030, 400048, 400060, 400004, 400039, 400017, 400018, 400029, 400069, 400051 | 正面與側面 |

**真實的部分**：照片（1334×2048）、相機校正、皮膚／眼睛／打光、相機間色差、
AliceVision 2023.3 的 meshing / meshFiltering / meshDenoising / texturing 二進位檔（CPU 版）、
本腳本自己的匯出程式碼（consensus、上採樣、EXR/SfM 寫出）與 v3 參數。
k=3，跟實際 1024 解析度的執行一樣。

**模擬的部分**：VGGT-Omega 的深度圖。從 Multiface 追蹤網格（細分 + 平滑後當參考曲面）
在網路解析度（416×640）渲染，再加上每個視角各自的低頻偏移（標準差 0.6 mm）和像素雜訊
（0.15 mm），模擬 VGGT 各視角彼此不一致的情況。

重現方式（資料集只下載一格影格；38 台相機的 PNG 從 3.1 GB 的 tar 串流中擷取，約 1.5 分鐘）：

```bash
B=https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/6795937
curl -sS $B/metadata.tar | tar -x
curl -sS $B/tracked_mesh--E001_Neutral_Eyes_Open.tar | tar -x
curl -sS $B/images--E001_Neutral_Eyes_Open.tar | tar -x --wildcards '*/000102.png'
python experiments/multiface_eval.py   --multiface m--20180227--0000--6795937--GHS --av-root $ALICEVISION_ROOT --work /tmp/mf_eval
python experiments/geometry_ablation.py --multiface m--20180227--0000--6795937--GHS --av-root $ALICEVISION_ROOT --work /tmp/mf_geom
```

**限制**：只有一位受試者、一格影格、一組模擬的深度誤差；數字代表方向，不是定論。
真正的驗收要在 `C:/atop/data/test_set.txt` 那 5 組上跑（§8.5）。

### 8.2 貼圖：`texture_pyramid.py`（Laplacian 金字塔 + 光流對齊）

**為什麼調 `aliceVision_texturing` 解不開**（接續 §7.5–§7.7）：AliceVision 以**三角形**為單位、
以**整數視角數**分配頻段，band 1 最少也要平均 2 個視角，最高頻段在三角形邊界硬切換；
而且視角是照網格投影的位置直接疊加，網格差 0.5–1 mm 時兩張照片會錯開 1–3 px。

**新做法**：AliceVision 貼圖後，保留它的 UV 展開，重新烘焙每個 texel：

1. **每個頻段用連續的權重銳度 p_b**（Burt & Adelson 1983）：權重在各視角自己的影像空間計算
   （入射角、解析度、離遮擋邊界／影像邊界的距離），取 p_b 次方後用 Gaussian 金字塔平滑到
   該頻段的尺度。p=8 ≈「幾乎只用最佳視角」，p=1 = 加權平均。預設 `8 6 4 2 1 1`（最細頻段在前）。
   權重在每個頻段的過渡寬度等於該頻段的波長，所以**任何頻段都沒有三角形邊界的接縫**。
2. **每張照片先對齊再混合**：把參考貼圖（以最佳視角為主的混合）渲染回該視角，用 DIS 光流
   找出網格預測的內容在照片裡實際在哪裡，texel 在修正後的位置取樣（Eisemann et al. 2008,
   *Floating Textures*）。實測各視角錯位中位數 0.3–1.0 px，p95 為 2–6 px，與 §7.6 的推算一致。
3. **逐表面點的增益**：用多個視角看到的**同一個表面點**，對多視角中位數做穩健比值，
   這才是 `--harmonize` 原本想做的事（§7.7 量到它對構圖敏感、實際無效）。
4. **鏡面高光降權**：某視角在同一點比多視角中位數亮很多時降低權重；只作用在第 2 頻段以上
   （高光本來就在低頻，§7.7），最細的兩個頻段留給幾何上最好的視角。

每個頻段是精確分解（Σ_b L_b = I），只有一個視角看得到的 texel 會被原樣重現。

**結果**（10 個保留視角平均；眼周 = 兩眼中心 14 mm 內；`_reg` = 先用光流消除網格造成的錯位再評分，
量到的是模糊／鬼影／接縫，而不是幾何誤差；細節能量比 1 = 跟照片一樣銳利）：

| 方法 | PSNR | PSNR_reg | 眼周 PSNR_reg | 眼周 SSIM_reg | 眼周細節能量比 | 眼周細節相關 |
|---|---|---|---|---|---|---|
| AliceVision v3（`1 1 10 0`） | 30.72 | 31.51 | 26.32 | 0.701 | 0.644 | 0.117 |
| 只取最佳視角 | 30.82 | 31.91 | 26.75 | 0.723 | 0.697 | 0.122 |
| 全部視角加權平均 | 31.40 | 32.32 | 27.26 | 0.729 | 0.523 | 0.120 |
| **金字塔 `8 6 4 2 1 1`（預設）** | 31.14 | 32.28 | **27.34** | **0.741** | **0.686** | **0.137** |
| 金字塔 `32 16 4 2 1 1`（更銳利） | 31.07 | 32.24 | 27.20 | 0.735 | **0.758** | 0.135 |

* 金字塔版本在眼周**每一個指標都贏 AliceVision v3**，而且兩端都顧到：比「加權平均」銳利
  （0.686 vs 0.523），又比「只取最佳視角」忠實（相關 0.137 vs 0.122）。
* 肉眼差異最大的是**接縫**：AliceVision v3 在鼻樑、內眼角有明顯的亮色色塊，金字塔版本沒有
  （3D 檢視器「貼圖」比較）。
* `--pyr-specular-min-band 0`（高光降權套用到所有頻段）更忠實但較軟：同一網格上眼周
  PSNR_reg 27.62、細節相關 0.161，細節能量比降到 0.548。
* 光流對齊的貢獻小但一致為正（同設定下眼周細節相關 0.152 → 0.161、中頻相關 0.522 → 0.535）。
  它的效果受限於殘留的幾何誤差，幾何越準、對齊越能讓中頻多平均幾個視角。

### 8.3 幾何：四個深度匯出修正（`--pixel-center` / `--consensus-*` / `--depth-upsample`）

1. **`--pixel-center aligned`（最重要）**。舊版把網路解析度的內參直接乘 k，但管線裡每一次縮放
   （PIL 縮照片、cv2 放大深度圖）都是中心對齊的：網路像素 i 的中心在影像像素 k·i + (k−1)/2，
   不是 k·i。k=3 時**每個視角的光線（深度與顏色一起）偏了 1 個影像像素**（臉上約 0.13 mm），
   而且每個視角偏的方向是它自己的影像軸，所以視角之間互相對不上。
2. **`--consensus-sampling bilinear`**：consensus 查鄰近視角深度時不再取最近像素，
   避免在斜面上產生規則的量化波紋。
3. **`--consensus-fuse inlier-mean`**：不再用中位數（它每個像素挑不同視角，造成 0.5 pixSize 的
   補丁狀階梯），改為平均中位數 ±2 pixSize 內的所有佐證值。
4. **`--depth-upsample cubic`**：雙三次插值是 C1 連續，不會把網路像素的格狀折面印進網格。

**結果**（臉部區域與參考曲面比較；法線誤差就是打光時看到的凹凸）：

| 設定 | 距離中位數 mm | 距離 p90 mm | 法線誤差中位數 | 法線誤差 p90 | 原始網格二面角 >10° |
|---|---|---|---|---|---|
| v3 | 0.098 | 0.270 | 1.93° | 6.04° | 22.5% |
| v3 + cubic | 0.097 | 0.269 | 1.92° | 5.99° | 24.3% |
| v3 + bilinear | 0.102 | 0.285 | 2.01° | 6.18° | 20.3% |
| v3 + inlier-mean | 0.100 | 0.256 | 1.86° | 5.88° | 23.9% |
| v3 + **aligned** | 0.088 | 0.249 | 1.90° | 5.85° | **10.6%** |
| **v4-depth（四項合併）** | **0.081** | **0.235** | **1.81°** | **5.67°** | **10.2%** |

（除了最後一欄，都是 meshDenoising 之後的網格。原始網格的法線誤差 v3 3.13° → v4 2.19°。）

單獨看只有 `aligned` 效果明顯，但四項合併比任何一項都好。

### 8.4 試過但**沒有**採用的（凹凸）

| 在 v4-depth 之上再加 | 距離中位數 | 距離 p90 | 法線誤差中位數 | 結論 |
|---|---|---|---|---|
| Taubin 平滑 ×20（`experiments/mesh_smooth.py`） | 0.080 | 0.232 | 1.83° | 無效，p90 二面角反而上升 |
| `--consensus-smooth-radius 4` | 0.080 | 0.232 | 1.86° | 變差，原始網格更粗糙 |
| `--denoise-iterations 5` | 0.095 | 0.273 | 2.19° | **看起來更平，但偏離真實曲面** |
| Meshroom 範本 denoise（λ2, η1.8, ν0.3）×5 | 0.121 | 0.383 | 2.70° | 看起來最平（二面角 >10° 只剩 0.1%），**誤差最大** |

這解釋了 §7.3(c)：Meshroom 那種「乾淨平滑」的觀感是把曲面壓成一片片平面換來的，
對術前模擬是錯的方向。denoise 之後剩下的起伏是**中頻**（幾 mm 到十幾 mm），來自各視角深度
彼此不一致，所以必須在網格化**之前**處理（§8.3），網格後處理只能把它抹平成錯的形狀。

### 8.5 v4 建議指令

```bash
python vggt_omega_to_alicevision.py \
  --images $D/photos --masks $D/masks_david --output $D/photos_alicevision_v4 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt --av-bin $ALICEVISION_ROOT/bin \
  --image-resolution 1024 --device cuda \
  --pixel-center aligned --consensus-sampling bilinear --consensus-fuse inlier-mean \
  --depth-upsample cubic \
  --retexture pyramid
# 輸出：$D/photos_alicevision_v4/texturedMesh_pyramid/texturedMesh.obj
#（AliceVision 原本的 texturedMesh/ 也會保留，方便對照）
```

已有 v3 輸出、只想換貼圖時（不重跑推論與 meshing）：

```bash
python texture_pyramid.py --input $D/photos_alicevision_v3 --debug
# 想要更銳利的皮膚紋理：--band-sharpness 32 16 4 2 1 1
# 想要最忠實於照片：   --specular-min-band 0
```

所有新選項預設都是關閉的（v3 行為），舊結果可以原樣重現。

**記憶體**：`texture_pyramid.py` 在 8192 圖集（約 4,000 萬 texel）時估計需要 5–6 GB RAM（未實測，4096 圖集實測可在 15 GB 的機器上跑）；
先用 `--texture-side 4096` 試跑。9 個視角、4096 圖集在 4 核 CPU 上約 2 分鐘。

### 8.6 其他發現

* **AliceVision 輸出的網格座標是 (x, −y, −z)**，也就是相對於 SfMData 相機（OpenCV 慣例）
  做了 `diag(1,−1,−1)`。`av_mesh_io.align_mesh_to_cameras()` 會自動偵測並轉回來；
  任何拿 AliceVision 網格去跟相機一起算的程式都要注意這點。
* 同一組深度圖重跑 meshing，法線誤差會有約 ±0.01° 的波動（graph-cut 的隨機性），
  小於這個量級的差異不要當真。

### 8.7 下一步

1. 在 `test_set.txt` 的 5 組真實 capture 上跑 §8.5 的 v4 指令，和 v3 並排看（3D 檢視器的
   `experiments/export_viewer_assets.py` 可以直接打包任何 OBJ）。
2. 幾何誤差變小之後，光流對齊能讓中頻平均更多視角：再掃一次 `--band-sharpness` 的中段（第 2–3 頻段）。
3. §7.9 的第 2 點（`--image-resolution 1152`）與 §7.9 的第 3 點（交叉偏振片）仍然成立，
   而且跟本節的改動互不衝突。
