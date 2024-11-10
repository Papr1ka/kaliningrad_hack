import faiss
import numpy as np

index = faiss.read_index("vacancy_index.index")


def find_simmilar(query, k=10):
    """
    Поиск похожих описаний

    Args:
      query (numpy array): Оценка OCEAN размерности (1, 5)
      k (int): Требуемое количество похожих описаний
    Returns:
      list: Список формата [{'ID': ид похожего описания в датасете с описаниями (int),
      'Score': процент похожести в диапазоне от 0 до 1 (float)}] длиной k
    """

    assert query.shape == (1, 5)

    distances, indices = index.search(query, k)

    result = []
    for i, idx in enumerate(indices[0]):
        result.append({'ID': idx, 'Score': round(1 / (1 + distances[0][i]), 3)})

    return result


def delete(remove_id, new_index_path='vacancy_index.index'):
    """
    Удаление из индекса по id

    Args:
      remove_id (int): Ид вектора, который нужно удалить
      new_index_path (str): Путь для сохранения индекса
    Returns: None
    """

    remove_set = np.array([remove_id])
    index.remove_ids(remove_set)
    faiss.write_index(index, new_index_path)


def add(query_id, query, new_index_path='vacancy_index.index'):
    """
    Добавление в индекс

    Args:
      query_id (int): Ид вектора, с которым он записывается в индекс
      query (numpy array): Оценка OCEAN размерности (1, 5)
      new_index_path (str): Путь для сохранения индекса
    Returns: None
    """

    assert query.shape == (1, 5)

    text_id = np.array([query_id])
    index.add_with_ids(query, text_id)
    faiss.write_index(index, new_index_path)
