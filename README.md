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
      kolomna-render  — рендер (wgpu): изометрия, тайлы, камера, текстурный массив террейна
      kolomna-app     — приложение (окно winit, мягкая камера, exe)

## Быстрый старт

    rustup default stable-x86_64-pc-windows-gnu   # на этой машине MSVC нет; C: забит — GNU-тулчейн
    cargo build

    # приложение (окно): мягкий зум колесом, повороты средней кнопкой, навигация краями окна
    cargo run -p kolomna-app -- "C:/Users/<user>/Kolomna/city.save" "data2.dat"

    # одиночный кадр без окна (для проверок)
    cargo run -p kolomna-app -- --shot out/city.png <city.save> <data2.dat>

    # субфайлы: листинг и извлечение (со сжатием)
    cargo run -p kolomna-dbpf --example entries -- <file.dat>
    cargo run -p kolomna-dbpf --example extract -- <file.dat> <type> <group> <instance> <out.bin>

    # текстуры FSH по TGI
    cargo run -p kolomna-assets --example fsh_pick -- "F:/.../data2.dat" out/citytex 7ab50e44 891b0e1a 13

    # рендер реального города из сейва в PNG (движок kolomna-render напрямую)
    cargo run -p kolomna-render --example city_terrain -- "C:/Users/<user>/Kolomna/city.save" "data2.dat" out/city.png

Без аргументов `kolomna-app` открывает город из собственных процедурных данных.

Готовый exe для запуска: **`Kolomna.exe` в корне** (это копия `target/release/kolomna-app.exe`; собрать заново: `cargo build --release`).

## Управление (окно)

- колесо — мягкий зум; средняя кнопка — мягкий поворот; Shift+средняя / ЛКМ / ПКМ — перенос
- края окна — плавная навигация; WASD/стрелки — перенос; Q/E — поворот на 90°; R — сброс; Esc — выход
- в заголовке окна: fps, зум, угол

## Статус

- [x] DBPF-ридер: заголовок (96 Б), индекс (20 Б/запись), поиск и чтение субфайлов
- [x] RefPack-декомпрессор + DIR-список сжатых (проверено: 200/200 случайных сжатых файлов data2)

- [x] FSH-декодер: DXT1/DXT3/32/24/16/8-бит → RGBA + дамп PNG (`--example fsh_dump`)
- [x] Изометрический рендер террейна из данных копии (offscreen → PNG: `--example terrain_preview`, `--example city_terrain`)
- [x] Сейвы: чтение террейна города (чанк `a9dd6ff4`: сетка 129×129 f32, метры) и рендер реального города
- [x] Текстуры террейна из архивных данных: массив 6×256×256, непрерывные UV с повтором (без «квадратов»), травы по высоте, песок — берега, суглинок — обрывы
- [x] Вода процедурная (нормали волн, оттенок от глубины, прибой); поверхность воды — по моде высот сейва (+2.2 м над полкой мелководья)
- [x] Окно (winit) + приложение `kolomna-app`: мягкий зум/повороты/краевая навигация, `--shot` для одиночного кадра, релизный exe

- [ ] Рендер города с постройками: 3DMD-парсер + EXMP (полоса застройки) + размещение из сейва
- [ ] Симуляция: спрос/рост/бюджет








| Текстуры | `data2..5.dat`, тип `7ab50e44` | FSH (декодер есть) |

## Референсы (только чтение, не вендоринг)

- DBPF/форматы: публичные описания форматов,
  open-source ридеры DBPF
- RefPack/QFS: [RefPack-QFS-Resources](https://github.com/lingeringwillx/Refpack-QFS-Resources),
  открытые исходники RefPack/QFS
- Родственные проекты: [scion](https://github.com/nsgomez/scion),
  [scgl](https://github.com/nsgomez/scgl), [gzcom-dll](https://github.com/nsgomez/gzcom-dll)
