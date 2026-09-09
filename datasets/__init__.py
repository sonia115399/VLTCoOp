from .tea import tea
from .corn import corn
from .datepalm import datepalm
from .allamanda_cathartica import allamanda_cathartica
from .duranta_erecta_gold import duranta_erecta_gold
from .excoecaria_cochinchinensis import excoecaria_cochinchinensis
from .ficus_microcarpa import ficus_microcarpa
from .gypsophila_paniculata import gypsophila_paniculata
from .liriope_muscari_variegata import liriope_muscari_variegata
from .murraya_exotica import murraya_exotica
from .ruellia_simplex import ruellia_simplex
from .mix import mix
from .mix1 import mix1

dataset_list = {
                "tea": tea ,
                "corn": corn ,
                "datepalm": datepalm ,
                "allamanda_cathartica": allamanda_cathartica ,
                "duranta_erecta_gold": duranta_erecta_gold ,
                "excoecaria_cochinchinensis": excoecaria_cochinchinensis ,
                "ficus_microcarpa": ficus_microcarpa ,
                "gypsophila_paniculata": gypsophila_paniculata ,
                "liriope_muscari_variegata": liriope_muscari_variegata ,
                "murraya_exotica": murraya_exotica ,
                "ruellia_simplex": ruellia_simplex,
                "mix": mix,
                "mix1": mix1
                }


def build_dataset(cfg):
    return dataset_list[cfg.DATASET.NAME](cfg)