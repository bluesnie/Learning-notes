###### datetime:2020/6/28 14:34
###### author:nzb

    
- 唯一验证
    ```python
  # serialiezers.py

    from rest_framework import serializers
    from rest_framework.validators import UniqueTogetherValidator
        
    class ExampleSerializer(serializers.ModelSerializer):
        args1 = serializers.CharField(label="参数1", required=False)
    
        class Meta:
            model = Example
            exclude = ['create_time']
            validators = [UniqueTogetherValidator(queryset=Example.objects.filter(), 
                                                  fields=('project_id', 'project_type', 'user_id'),
                                                  message='已存在')]
        def validate(self, attrs):
            file = attrs.get("file", None)
            # 文件类型验证
            if validate_file(file) == "unknown":
                raise serializers.ValidationError(code=40000, detail="不支持的文件类型")
    
            return attrs
    ```
    
- 文件类型验证

```python
    # helper.py
    
    import struct
    
    # 常见文件格式的文件头
    ALLOW_FILETYPE = {
            "FFD8FF": "JPEG (jpg)",
            "89504E47": "PNG (png)",
            "47494638": "GIF (gif)",
            "49492A00": "TIFF (tif)",
            "41433130": "CAD (dwg)",
            "D0CF11E0": "MS Word/Excel (xls.or.doc)",
            "255044462D312E": "Adobe Acrobat (pdf)",
            "504B0304": "ZIP Archive (zip)",
            "52617221": "RAR Archive (rar)",
            "41564920": "AVI (avi)"
    }

    # 字节码转16进制字符串
    def bytes2hex(bytes):
        num = len(bytes)
        hexstr = u""
        for i in range(num):
            t = u"%x" % bytes[i]
            if len(t) % 2:
                hexstr += u"0"
            hexstr += t
        return hexstr.upper()
    
    def validate_file(file):
        """
        根据文件头判断文件类型
        文件后缀不可信，并且后缀在linux系统下是没有这个概念的，所以通过文件中的头部标识来判断
        :param file:IO文件
        :return: xxx：文件类型，unknown：未知文件（不支持）
        """
        # binfile = open(file, 'rb')  # 必需二制字读取
        tl = ALLOW_FILETYPE
        ftype = 'unknown'
        for hcode in tl.keys():
            numOfBytes = len(hcode) / 2  # 需要读多少字节
            file.seek(0)  # 每次读取都要回到文件头，不然会一直往后读取
            # hbytes = struct.unpack_from("B" * numOfBytes, binfile.read(numOfBytes))  # 一个 "B"表示一个字节
            hbytes = struct.unpack_from("B" * int(numOfBytes), file.read(int(numOfBytes)))  # 一个 "B"表示一个字节
            f_hcode = bytes2hex(hbytes)
            if f_hcode == hcode:
                ftype = tl[hcode]
                break
        file.seek(0)        # 回到文件头
        return ftype

```