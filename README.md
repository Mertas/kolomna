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
    # субфайлы: листинг и извлечение (со сжатием)
    cargo run -p kolomna-dbpf --example entries -- <file.dat>
    cargo run -p kolomna-dbpf --example extract -- <file.dat> <type> <group> <instance> <out.bin>
    # текстуры FSH по TGI
    cargo run -p kolomna-assets --example fsh_pick -- "F:/.../data2.dat" out/citytex 7ab50e44 891b0e1a 13
    # рендер реального города из сейва в PNG
    cargo run -p kolomna-render --example city_terrain -- "C:/Users/<user>/Kolomna/city.save" "data2.dat" out/city.png

## Статус

- [x] DBPF-ридер: заголовок (96 Б), индекс (20 Б/запись), поиск и чтение субфайлов
- [x] RefPack-декомпрессор + DIR-список сжатых (проверено: 200/200 случайных сжатых файлов data2)

- [x] FSH-декодер: DXT1/DXT3/32/24/16/8-бит → RGBA + дамп PNG (`--example fsh_dump`)
- [x] Изометрический рендер террейна из данных копии (offscreen → PNG: `--example terrain_preview`, `--example city_terrain`)
- [x] Сейвы: чтение террейна города (чанк `a9dd6ff4`: сетка 129×129 f32, метры) и рендер реального города
- [ ] Окно (winit) + свободная камера
- [ ] Сейвы: зоны, сети, здания
- [ ] Симуляция: спрос/рост/бюджет

## Референсы (только чтение, не вендоринг)

- DBPF/форматы: публичные описания форматов,
  open-source ридеры DBPF
- RefPack/QFS: [RefPack-QFS-Resources](https://github.com/lingeringwillx/Refpack-QFS-Resources),
  открытые исходники RefPack/QFS
- Родственные проекты: [scion](https://github.com/nsgomez/scion),
  [scgl](https://github.com/nsgomez/scgl), [gzcom-dll](https://github.com/nsgomez/gzcom-dll)
