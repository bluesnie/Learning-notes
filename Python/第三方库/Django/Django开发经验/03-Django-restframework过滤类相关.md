###### datetime:2020/5/13 13:45
###### author:nzb

## 过滤类相关

```python

class NumberInFilter(django_filters.BaseInFilter, django_filters.NumberFilter):
    pass

class MyFilter(django_filters.rest_framework.FilterSet):
    id_list = NumberInFilter(field_name="id", label="xxx", lookup_expr="in")

    class Meta:
        model = TestModel
        fields = ['id_list']
```
