import pytest
from mpa.modules.datasets.det_csv_dataset import CSVDatasetDet

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_det_csv_dataset(tmpdir):
    data_csv = tmpdir.join('data.csv')
    anno_csv = tmpdir.join('anno.csv')
    data_csv.write("""
ImageID,ImagePath,Width,Height
0,Shoppertrack_3983_Reidentification_Shoppertrack_0_Cam1cut1.mp4_00001723.jpg,1280,720
1,Shoppertrack_3984_Reidentification_Shoppertrack_0_Cam2cut1.mp4_00002286.jpg,1280,720
2,Shoppertrack_3984_Reidentification_Shoppertrack_0_Cam2cut1.mp4_00003772.jpg,1280,720
3,Shoppertrack_3983_Reidentification_Shoppertrack_0_Cam1cut1.mp4_00003659.jpg,1280,720
    """)

    anno_csv.write("""
AnnoID,ImageID,Xmin,Ymin,Xmax,Ymax,ClassName,IsOccluded
0,0,317.989990234375,55.9900016784668,389.0799865722656,142.0,Person,True
1,0,299.8999938964844,112.30000305175781,394.0199890136719,365.70001220703125,Person,False
2,0,791.8499755859375,357.75,977.5999755859375,715.9500122070312,Person,False
3,0,631.0,127.9000015258789,779.5999755859375,416.6000061035156,Person,False
4,0,456.7200012207031,66.05000305175781,528.5999755859375,202.8000030517578,Person,True
5,0,410.29998779296875,115.30000305175781,517.25,286.79998779296875,Person,False
8,3,294.29998779296875,77.5999984741211,366.29998779296875,149.39999389648438,Person,True
9,3,728.7999877929688,117.80000305175781,815.9000244140625,363.3999938964844,Person,False
10,3,260.70001220703125,158.6300048828125,390.0299987792969,474.3999938964844,Person,False
11,3,0.10000000149011612,506.70001220703125,31.700000762939453,571.0999755859375,Person,True
12,3,183.02999877929688,142.0800018310547,281.2799987792969,409.04998779296875,Person,False
13,3,452.6000061035156,62.810001373291016,527.7000122070312,201.39999389648438,Person,True
14,3,420.8299865722656,112.41999816894531,522.5999755859375,286.29998779296875,Person,False
    """)

    dataset = CSVDatasetDet(
        data_file=data_csv,
        ann_file=anno_csv,
        pipeline=[]
    )

    assert len(dataset.data_infos) == 2
    assert len(dataset.data_infos[0]['ann']['bboxes']) == 4  # 6 - 2 (occluded)
    assert len(dataset.data_infos[1]['ann']['bboxes']) == 4  # 7 - 3 (occluded)
