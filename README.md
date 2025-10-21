
## RGBA Convert
`dataset_tool.py` 모듈 최대한 활용.
* 굳이 따로 RGBA converter 만들필요 없음.
* transform 하기 전 alpha 채널 추가해줘야 흰색 배경이 깔끔하게 alpha 채널로 분리됨 (interpolation 때문에 픽셀값 바뀜)
* 흰색 부분을(threshold=5 포함) alpha channel로 판단하도록 했지만 찌르르공같이 흰색이 `(255, 255, 255)` foreground에 포함되어있으면 alpha 채널로 변환함. 

```python
def get_alpha_channel(img: np.ndarray, resolution:Tuple[int, int], thresh: int = 5) -> np.ndarray:
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, c = img.shape
    if c == 4:
        return img
    if c not in (1, 3):
        raise ValueError(f"unsupported channels: {c}")
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {img.dtype}")

    if c == 1:
        white_mask = img[:, :, 0] >= 255 - thresh
    else:
        white_mask = np.all(img >= 255 - thresh, axis=2)

    alpha = np.where(white_mask, 0, 255).astype(np.uint8)
    pil_alpha = PIL.Image.fromarray(alpha, mode="L")
    pil_alpha = pil_alpha.resize(resolution, PIL.Image.LANCZOS)
    return np.array(pil_alpha)
```


```bash
python dataset_tool.py --source=./data/pokemon --dest=./data/pokemon256.zip \
  --resolution=256x256 --transform=center-crop --to_rgba
```


## RGBA Image 학습
Generator는 이미지 channel 크기를 자동으로 반영함.
하지만 Discriminator는 따로 img_channel 입력안받고 3으로 가정함.
명시적으로 img_channel 에 따라 첫 conv in_channel 세팅하도록 수정

```python
#pg_modules/discriminator.py
class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.feature_network = F_RandomProj(**backbone_kwargs, **kwargs) # in_channel 정보가 담긴 kwargs 인자로 전달
        ...



# pg_modules/projector.py
def _make_projector(im_res, cout, proj_type, expand=False, img_channel=3):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True, in_chans=img_channel)
```

그 외 기타 [1, 3] 채널만 검사하거나 동작하는 부분 4채널인 케이스 추가. 

## 최종 학습
```bash
python train.py --outdir ./training-runs/ --cfg fastgan --data ./data/pokemon_RGBA.zip --gpus 1 --batch 8 --mirror 1 --snap 50 --batch-gpu 8 --kimg 10000
```