"""Produce a portable HTML experiment report from cached detections."""
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np
from probe import cluster_frames

OUT = Path(__file__).resolve().parent
results = json.loads((OUT / 'results.json').read_text())
findings = {
    'clip_1': 'На просмотренном обзорном листе основная вебка находится сверху слева. YuNet: главный кластер 48/48; случайные лица встречаются по одному разу. Haar пропускает часть кадров.',
    'clip_2': 'Основная вебка снизу справа. YuNet: 47/48 в главном позиционном кластере. Уточнение второго эксперимента: #7 относится к другой сцене/просматриваемому клипу, а не подтверждённой текущей вебке. Поэтому 47/48 не означает 47 правильных наблюдений ведущего. На последнем сэмпле (5356.6 с) ведущий отвернулся и держит руку у лица; панель остаётся. Дополнительный кластер 4/48 соответствует лицу игрового персонажа.',
    'clip_3': 'Вебка снизу слева; большая часть кадра — другая камера того же человека. Все 48 сэмплов просмотрены на контактных листах. Главный кластер YuNet 34/48; ещё 5/48 — тот же ведущий внутри той же панели, но со смещением головы. Итого вручную сопоставлено 39/48 кадров с детекцией лица в нужной панели. Уточнение второго эксперимента: #6 содержит меньшую почти чёрную панель (раньше ошибочно считалась отсутствующей); #7 — монтажный крупный план без подтверждённой панели; #47 — финальная заставка. В #22–25, #27, #35 панель остаётся, но лицо перекрыто, отвернуто или не найдено. Haar также ошибочно находит лица на контроллере и предметах.'
}
parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Проверка на трёх видео</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;background:#fafafa;color:#222}img{max-width:100%}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccc;padding:9px;text-align:left}section{background:white;padding:20px;margin:24px 0;border-radius:12px}code{background:#eee}summary{cursor:pointer}</style><h1>Проверка на трёх реальных видео</h1><p><b>Эксперимент, не готовый детектор панели.</b> 48 равномерных midpoint-кадров на каждый ролик, 144 всего. Зелёные боксы — YuNet, оранжевые — Haar. Это боксы лиц, не границы вебки. Облачные vision API не использовались.</p><p>Повторяемость оценивается простым глобальным объединением по положению и размеру. Она не доказывает роль человека. Разные разрешения и разные ролики не позволяют сравнивать скорость или сложность видео напрямую.</p>']
parts.append('<p><b>Исторический первый эксперимент.</b> Исправления по clip_3 #6 и clip_2 #7 внесены после независимой разметки. <a href="panel-report.html">Актуальный отчёт: границы панелей и проверка без лица.</a></p>')
summary = []
for v in results['videos']:
    video_id = v['id']
    parts.append(f'<section><h2>{html.escape(v["id"])}</h2><p>{html.escape(video_id)} · {v["duration_s"]/60:.1f} мин · {v["width"]}×{v["height"]} · шаг {v["duration_s"]/48:.1f} с</p><p>{findings[video_id]}</p>')
    parts.append('<h3>Чувствительность к числу кадров</h3><p>Подвыборки из кэша: n равномерно распределённых кадров, индексы floor((i+0.5)×48/n). Это один фиксированный вариант выборки, не статистическая оценка надёжности. При n=3 позиции другие, чем в неизвестном исходном опыте из переписки. В ячейке — размер крупнейшего пространственного кластера; это не recall панели.</p><table><tr><th>Кадров</th><th>Haar</th><th>YuNet</th></tr>')
    comparison = []
    for n in [3, 12, 24, 48]:
        indices = ((np.arange(n) + 0.5) * 48 / n).astype(int)
        subset = [v['frames'][int(i)] for i in indices]
        row = {'n': n}
        for detector in ['haar', 'yunet']:
            clusters = cluster_frames(subset, detector)
            row[detector] = clusters[0]['count'] if clusters else 0
        comparison.append(row)
        parts.append(f'<tr><td>{n}</td><td>{row["haar"]}/{n}</td><td>{row["yunet"]}/{n}</td></tr>')
    parts.append('</table>')
    t = v['timing_s']
    parts.append(f'<p>Обработка 48 кадров: {t["total"]:.2f} с, включая decode {t["decode"]:.2f}, Haar {t["haar"]:.2f}, YuNet {t["yunet"]:.2f}. Без скачивания и установки зависимостей; один локальный прогон, не бенчмарк real-time.</p>')
    image = base64.b64encode((OUT / f'{video_id}-overview.jpg').read_bytes()).decode()
    parts.append(f'<h3>Обзор: каждый четвёртый сэмпл</h3><img alt="Обзор детекций" src="data:image/jpeg;base64,{image}">')
    for page in range(1, 5):
        image = base64.b64encode((OUT / f'{video_id}-page{page}.jpg').read_bytes()).decode()
        parts.append(f'<details><summary>Все сэмплы: страница {page}/4</summary><img alt="Страница {page}" src="data:image/jpeg;base64,{image}"></details>')
    parts.append('</section>')
    summary.append({'id': video_id, 'comparison': comparison})
parts.append('<section><h2>Что меняется в плане</h2><ol><li>YuNet — основной кандидат на face detector; Haar оставить только для сравнения. В этой среде веса скачались.</li><li>Сначала лицо предлагает область поиска, затем нужно находить и сопровождать панель. Не группировать всю вебку только по центру головы.</li><li>Не вводить глобальный порог 80% для целого смонтированного ролика. Отдельно учитывать присутствие панели, видимость лица и смены раскладки.</li><li>Добавлять плотные локальные окна вокруг пропусков и переходов. 48 кадров за 90 минут не проверяют краткие события.</li><li>Реализовать границы панели и проверить их по независимой разметке. Сейчас корректный JSON с границами всей вебки ещё не получен.</li></ol><h2>Ограничения</h2><p>Три выбранных примера не являются независимым тестовым набором. Нет количественной разметки границ и IoU, проверки отрицательных роликов и двух постоянно видимых конкурирующих окон. Число кадров само по себе не гарантирует улучшение: таблицы показывают результат одной конкретной подвыборки. 39/48 в clip_3 — ручное объединение двух кластеров после просмотра, не результат автоматической детекции панели.</p><h2>Воспроизведение</h2><pre>.venv/bin/python experiment/probe.py\n.venv/bin/python experiment/summarize.py</pre><p>Видео находятся в корне проекта; веса — experiment/yunet.onnx. Версии: opencv-python-headless==4.13.0.92, numpy==2.5.3. YuNet 2023mar из OpenCV Zoo; SHA256 сохранён в results.json. Параметры и сырые результаты детекций также в results.json.</p><h2>W8 — вопросы для ревью перед будущим merge</h2><ol><li>Почему 47/48 не означает 98% точности определения панели?</li><li>Почему в clip_3 основной кластер 34/48, но вручную найдено 39 кадров с нужным лицом?</li><li>Можно ли продолжать bbox панели при исчезновении лица и какие дополнительные проверки нужны?</li></ol><p>Merge не выполнялся. Это отчёт об исследовательском прогоне, а не утверждение готовности к production.</p></section></html>')
(OUT / 'report.html').write_text(''.join(parts))
(OUT / 'sampling-comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
# Compact illustrative montage: one sampled frame per video, no inferred panel boxes.
tiles = []
for v in results['videos']:
    image = cv2.imread(str(OUT / f'{v["id"]}-00.jpg'))
    h, w = image.shape[:2]
    for b in v['frames'][0]['yunet']:
        x, y, bw, bh = np.rint(np.array(b[:4]) * [w, h, w, h]).astype(int)
        cv2.rectangle(image, (x, y), (x+bw, y+bh), (60, 230, 60), 2)
    image = cv2.resize(image, (854, 480))
    tile = np.zeros((515, 854, 3), dtype=np.uint8)
    tile[:480] = image
    cv2.putText(tile, f'{v["id"]} @ {v["frames"][0]["actual_s"]:.1f}s - face boxes only', (10, 504), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    tiles.append(tile)
cv2.imwrite(str(OUT / 'preview.jpg'), np.vstack(tiles))
