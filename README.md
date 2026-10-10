# Kolomna

Градостроительный симулятор на Rust.
Рендер на wgpu и городской редактор в собственном коде.

## Правовая рамка (коротко)

- Код — только собственный, чистый Rust.
- Всё генерируется процедурно в коде.
- Имя, код и архитектура — свои.

## Структура

    crates/
      kolomna-dbpf    — чтение архивов DBPF (data*.dat, сейвы), RefPack-декомпрессор
      kolomna-assets  — декодеры форматов: FSH (текстуры), 3DMD (модели), RUL, Exemplar
      kolomna-render  — рендер (wgpu): изометрия, тайлы, камера
      kolomna-app     — приложение (окно, UI, запуск)

## Быстрый старт

    rustup default stable-x86_64-pc-windows-gnu   # на этой машине MSVC нет; C: забит — GNU-тулчейн
    cargo build
    cargo run -p kolomna-dbpf --example scan -- "data1.dat"

## Статус

- [x] DBPF-ридер: заголовок (96 Б), индекс (20 Б/запись), поиск и чтение субфайлов

- [ ] FSH-декодер → текстуры в RGBA
- [ ] Окно + изометрический рендер террейна из данных копии
- [ ] Сейвы: чтение города (террейн, зоны)
- [ ] Симуляция: спрос/рост/бюджет

## Референсы (только чтение, не вендоринг)

- DBPF/форматы: публичные описания форматов,
  open-source ридеры DBPF
- RefPack/QFS: [RefPack-QFS-Resources](https://github.com/lingeringwillx/Refpack-QFS-Resources),
  открытые исходники RefPack/QFS
- Родственные проекты: [scion](https://github.com/nsgomez/scion),
  [scgl](https://github.com/nsgomez/scgl), [gzcom-dll](https://github.com/nsgomez/gzcom-dll)
