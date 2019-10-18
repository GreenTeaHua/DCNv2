## Deformable Convolutional Networks V2 with Pytorch 1.3
win10+cuda10.1+python3.7+pytorch 1.3
### 4 build
  
  1. vs2015不行，模板问题，需要新的编译器，这里在线安装 vs2019的工具链（vs_buildtools__872086520.1559730552.exe）。    
  2. 启动 x64 Native Tools Command Prompt for VS 2019     
  3. conda （pip）安装 cudakit cudnn pytorch torchvision  
  4. pip list 检查相关包是否ok             
  5. python setup.py build develop # build  
  6. python test.py    # run examples and gradient check  


### An Example
- deformable conv
```python
    from dcn_v2 import DCN
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    output = dcn(input)
    print(output.shape)
```
- deformable roi pooling
```python
    from dcn_v2 import DCNPooling
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    # wrap all things (offset and mask) in DCNPooling
    dpooling = DCNPooling(spatial_scale=1.0 / 4,
                         pooled_size=7,
                         output_dim=32,
                         no_trans=False,
                         group_size=1,
                         trans_std=0.1).cuda()

    dout = dpooling(input, rois)
```
### Note
Now the master branch is for pytorch 1.0 (new ATen API), 
    
