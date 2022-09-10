from util import Utils


def test_sort_dict_by_value():
    input_dict = {1: 2, 2: 1, 3: 3}
    assert Utils.sort_dict_by_value(input_dict) == {2: 1, 1: 2, 3: 3}


def test_calculate_tf_values():
    sentence = ['oko', 'nos', 'oko', 'ucho', 'ucho']
    res = Utils.calculate_tf_values(sentence)
    assert len(res) == 3
    assert res['oko'] == 2
    assert res['ucho'] == 2
    assert res['nos'] == 1
