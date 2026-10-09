"""Evaluate panel predictions against separate manual development references."""
from collections import Counter
from pathlib import Path
import base64
import html
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiment'


def iou(a, b):
    x = max(a[0],b[0]); y = max(a[1],b[1])
    right = min(a[0]+a[2],b[0]+b[2]); bottom = min(a[1]+a[3],b[1]+b[3])
    intersection = max(0,right-x)*max(0,bottom-y)
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)


def edge_error(a,b):
    return max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[0]+a[2]-b[0]-b[2]),abs(a[1]+a[3]-b[1]-b[3]))


def evaluate(video, annotations):
    rows = {f['index']:f for f in video['frames']}
    details = []
    for ref in annotations:
        f = rows[ref['index']]
        actual, predicted = ref['bbox_px'], f['bbox_px']
        if actual is None:
            outcome = 'false_positive' if predicted else 'correct_abstention'
        elif predicted is None:
            outcome = 'miss_or_abstention'
        else:
            outcome = 'localized' if iou(actual,predicted) >= 0.9 else 'wrong_bounds'
        details.append({'index':ref['index'],'actual_s':f['actual_s'],'reference_bbox_px':actual,
                        'predicted_bbox_px':predicted,'status':f['status'],'reason':f['reason'],
                        'outcome':outcome,'iou':iou(actual,predicted) if actual and predicted else None,
                        'max_edge_error_px':edge_error(actual,predicted) if actual and predicted else None,
                        'reference_uncertainty_px':ref['uncertainty_px'],'reference_confidence':ref.get('confidence')})
    positive = [r for r in details if r['reference_bbox_px'] is not None]
    negative = [r for r in details if r['reference_bbox_px'] is None]
    localized = [r for r in positive if r['iou'] is not None]
    summary = {'references':len(details),'present':len(positive),'absent':len(negative),
               'outcomes':dict(Counter(r['outcome'] for r in details)),
               'mean_iou_when_returned':float(np.mean([r['iou'] for r in localized])) if localized else None,
               'mean_iou_including_misses_as_zero':float(np.mean([r['iou'] or 0 for r in positive])) if positive else None,
               'max_edge_error_px':max((r['max_edge_error_px'] for r in localized),default=None)}
    return {'summary':summary,'samples':details}


def embedded(path, max_width=1440):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Missing image {path}')
    if image.shape[1] > max_width:
        image = cv2.resize(image,(max_width,round(image.shape[0]*max_width/image.shape[1])))
    ok, jpg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(jpg).decode()


def comparison_preview(videos, references):
    selected = [('clip_1',0),('clip_2',47),('clip_3',24),
                ('clip_2',7),('clip_3',6),('clip_3',47)]
    canvas = np.zeros((840,1920,3),np.uint8)
    for i,(video_id,index) in enumerate(selected):
        v = videos[video_id]; f = v['frames'][index]
        cap = cv2.VideoCapture(str(ROOT/(video_id+'.mp4')))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC,f['actual_s']*1000)
            ok,image = cap.read()
            if not ok:
                raise RuntimeError('Cannot render comparison')
        finally:
            cap.release()
        image = cv2.resize(image,(640,360))
        reference = references[(video_id,index)]['bbox_px']
        for box,color,thickness in [(reference,(255,200,0),3),(f['bbox_px'],(40,240,40),1)]:
            if box:
                x,y,w,h = np.rint(np.array(box)*[640/v['width'],360/v['height'],640/v['width'],360/v['height']]).astype(int)
                cv2.rectangle(image,(x,y),(x+w-1,y+h-1),color,thickness)
        tile = np.zeros((420,640,3),np.uint8); tile[:360] = image
        cv2.putText(tile,f"{video_id} #{index} {f['actual_s']:.1f}s {f['status']}",(5,382),cv2.FONT_HERSHEY_SIMPLEX,0.49,(255,255,255),1)
        cv2.putText(tile,f['reason'],(5,407),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
        y,x = (i//3)*420,(i%3)*640; canvas[y:y+420,x:x+640] = tile
    cv2.imwrite(str(OUT/'panel-preview.jpg'),canvas)


def main():
    results = json.loads((OUT/'panel-results.json').read_text())
    reference = json.loads((OUT/'panel-reference.json').read_text())
    videos = {v['id']:v for v in results['videos']}
    refs = {(r['video_id'],r['index']):r for r in reference['annotations']}
    assert len(refs) == len(reference['annotations']), 'Duplicate references'
    evaluation = {v['id']:evaluate(v,[r for r in reference['annotations'] if r['video_id']==v['id']]) for v in videos.values()}
    (OUT/'panel-evaluation.json').write_text(json.dumps(evaluation,indent=2))
    comparison_preview(videos,refs)
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Панели вебки — эксперимент 2</title><style>body{font:17px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#222;background:#f6f7f9}section{padding:22px;background:white;border:1px solid #ddd;border-radius:10px;margin:22px 0}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #ddd;padding:8px;text-align:left}summary{cursor:pointer}pre{white-space:pre-wrap}.warning{border-left:5px solid #c80;padding-left:16px}</style><h1>Теперь проверяем всю панель, а не лицо</h1><p>W7: результат второго эксперимента. Исходная разреженная выборка и дополнительные кадры вокруг спорных моментов; количество указано ниже для каждого видео. Offline, три development-видео; не универсальная production-детекция.</p>',
             '<p>На сравнении ниже <b>голубая рамка — независимая ручная разметка, зелёная — предсказание</b>. При отказе зелёной рамки нет. Тёмная маленькая панель в clip_3 — реальный оставшийся пропуск.</p>',
             f'<img alt="Сравнение панелей и отказов" src="{embedded(OUT/"panel-preview.jpg")}">',
             '<section><h2>Как работает</h2><ol><li>Повторяющиеся лица предлагают угол кадра. При конкурирующих кластерах — отказ; это эвристика, не понимание роли ведущего.</li><li>Две внутренние прямые стороны прямоугольника ищутся по цветовым границам; ещё две стороны должны совпадать с краями кадра. Медиана поддержки по времени даёт основную геометрию.</li><li>На каждом сэмпле стороны ищутся заново. Допускается изменение размера в пределах ±25%; переход требует более сильных границ.</li><li>Содержимое ROI сравнивается с банком уменьшенных RGB-образцов. Это грубый appearance-фильтр, не распознавание личности. Текущий исходный кадр исключён из собственного сравнения.</li><li>Если лицо пропало, но текущие границы и содержимое подтверждают панель, bbox остаётся. Если свидетельств мало — ambiguous/not_detected и webcam=null. Старый bbox не переносится вслепую.</li></ol><p>Банк и основная геометрия строятся по всему ролику, в том числе будущим кадрам. Это не live-трекер. Дополнительные сэмплы: ±1/±2 с около максимум шести проблемных исходных кадров на видео; они не входят в обучение банка.</p></section>',
             '<section><h2>Независимая разметка и метрики</h2><p>50 сэмплов размечены отдельным проходом без просмотра предсказаний. 47 содержат целевую панель, 3 — нет. Допуск разметки обычно ±3 px; один слабый край в clip_2 — ±6 px. Разметчик один, часть кадров просмотрена контактными листами. Это независимое измерение границ на development-примерах, а не независимый test set: алгоритм и пороги разрабатывались на этих же роликах.</p><p>IoU — пересечение/объединение площади рамок. Localized означает IoU≥0.90; отдельно показаны пропуски, отказы и ложные bbox на отсутствии. Ambiguous на отсутствующей панели считается безопасным отказом, не доказанным распознаванием отсутствия.</p><table><tr><th>Видео</th><th>Верно локализовано / панель есть</th><th>Отказ / панель есть</th><th>Ложный bbox / панели нет</th><th>Средний IoU выданных</th><th>IoU с нулями за пропуски</th><th>Макс. ошибка стороны</th></tr>']
    for v in videos.values():
        s = evaluation[v['id']]['summary']; c = s['outcomes']
        parts.append(f'<tr><td>{html.escape(v["id"])}</td><td>{c.get("localized",0)}/{s["present"]}</td><td>{c.get("miss_or_abstention",0)}</td><td>{c.get("false_positive",0)}/{s["absent"]}</td><td>{s["mean_iou_when_returned"]:.3f}</td><td>{s["mean_iou_including_misses_as_zero"]:.3f}</td><td>{s["max_edge_error_px"]} px</td></tr>')
    parts.append('</table><p>Доля detected среди всех 48 сэмплов не является accuracy: не все 144 кадра размечены. Дополнительные кадры не имеют независимой количественной разметки.</p></section>')
    findings = {
        'clip_1':'Основная панель сверху слева. На разметке все 12 проверенных границ восстановлены. Стабильная геометрия устойчива даже когда сильнейший одиночный прямоугольник на кадре принадлежит интерфейсу.',
        'clip_2':'На #47 (5356.6 с) лицо не найдено, но панель определяется. На #32 (3671.2 с) верхняя граница слишком слаба: правильный консервативный статус — ambiguous, однако по разметке это пропуск существующей панели. На #7 (847.2 с) картинка из просматриваемого клипа/превью не принимается за текущую вебку из-за смены содержимого. Контекст ±2 с показывает возврат обычной вебки.',
        'clip_3':'Геометрия в начале отличается по высоте от основной части; локальный поиск учитывает это. На #22–25, #27 и #35 панель восстановлена без детекции лица — это шесть исходных кадров. На #6 (157.4 с) видна меньшая почти чёрная панель: границы-кандидаты найдены, но appearance-фильтр отказывается из-за сильного изменения содержимого. Это оставшийся пропуск, не отсутствие вебки. На #7 нет подтверждённой панели, на #47 финальная заставка: bbox не выдаётся.'}
    for v in videos.values():
        video_id = v['id']; counts = dict(Counter(f['status'] for f in v['frames']))
        faceless = sum(f['status']=='detected' and not f['face_in_candidate'] for f in v['frames'])
        parts.append(f'<section><h2>{html.escape(v["id"])}</h2><p>{findings[video_id]}</p><p>Исходные 48: {html.escape(str(counts))}; detected без найденного лица: {faceless}. Дополнительных кадров: {len(v["dense_frames"])}. Время этой стадии: {v["timing_s"]:.2f} с (включает декодирование/отчётные JPG, но не предыдущую детекцию лиц на исходных кадрах).</p><p>На следующих листах зелёная рамка — detected с лицом внутри, жёлтая — detected без лица. Рамка всегда панель, не bbox лица.</p>')
        parts.append(f'<img alt="Обзор панелей" src="{embedded(OUT/f"{video_id}-panel-overview.jpg")}">')
        for page in range(1,5):
            parts.append(f'<details><summary>Исходные сэмплы: {page}/4</summary><img alt="Все панели" src="{embedded(OUT/f"{video_id}-panel-page{page}.jpg")}"></details>')
        for page in range(1,(len(v['dense_frames'])+11)//12+1):
            parts.append(f'<details><summary>Дополнительные сэмплы: {page}</summary><img alt="Локальная проверка" src="{embedded(OUT/f"{video_id}-panel-dense{page}.jpg")}"></details>')
        parts.append('<details><summary>Все сравнения с разметкой</summary><table><tr><th>#</th><th>Эталон xywh</th><th>Прогноз xywh</th><th>IoU</th><th>Результат</th></tr>')
        for r in evaluation[video_id]['samples']:
            parts.append(f'<tr><td>{r["index"]}</td><td>{r["reference_bbox_px"]}</td><td>{r["predicted_bbox_px"]}</td><td>{round(r["iou"],3) if r["iou"] is not None else "—"}</td><td>{r["outcome"]}</td></tr>')
        parts.append('</table></details></section>')
    parts.append('<section class="warning"><h2>Исправления к первому отчёту</h2><ul><li>Утверждение «clip_3 #6 и #7 без панели» неверно: #6 содержит тёмную панель другого размера. Отсутствие лица не равно отсутствию камеры.</li><li>47/48 в кластере clip_2 не означает 47 правильных наблюдений текущего ведущего: #7 попал туда по позиции, но принадлежит другой сцене. Этот пример показывает проблему семантической идентификации.</li></ul></section>')
    parts.append('<section><h2>Что ещё не решено</h2><ul><li>Только прямоугольники, прикреплённые к углу. Не поддержаны свободно плавающие панели, круги, chroma key и несколько равноправных ведущих.</li><li>Резкая смена освещения/вида может дать отказ, как тёмная панель clip_3. Похожее превью теоретически может пройти appearance-проверку.</li><li>Нет непрерывных временных сегментов и точности момента смены сцены. Между редкими сэмплами ничего не гарантируется.</li><li>Результат не переносится автоматически на другие источники. Нужны новые ролики, отрицательные примеры и конкурирующие постоянные окна.</li><li>Пороги являются development-эвристиками. Score не вероятность. Даже хорошие IoU на 50 сэмплах не доказывают production-качество.</li></ul><h2>Артефакты и воспроизведение</h2><pre>.venv/bin/python experiment/panel_probe.py\n.venv/bin/python experiment/panel_report.py\n.venv/bin/python -m unittest discover -s experiment -p "test_panel_probe.py"</pre><p>panel-results.json — предсказания, debug-признаки и параметры; panel-reference.json — отдельная ручная разметка; panel-evaluation.json — сравнения; panel_probe.py не читает разметку. JSON bbox_px: x,y,width,height с исключающими правой/нижней границами; webcam — нормализованные x/y/w/h. При отказе оба поля null.</p><h2>W8 — короткая проверка понимания перед будущим merge</h2><ol><li>Почему панель можно подтвердить без детекции лица, но нельзя просто бесконечно сохранять прошлый bbox?</li><li>Чем ambiguous на тёмной панели отличается от вывода «вебки нет»?</li><li>Почему этот offline-эксперимент нельзя объявить готовым live-трекером или независимым benchmark?</li></ol><p>Merge не выполнялся. Следующий полезный этап — проверка замороженных правил на новых источниках, а не подгонка под эти три.</p></section></html>')
    (OUT/'panel-report.html').write_text(''.join(parts))
    print(json.dumps({k:v['summary'] for k,v in evaluation.items()},indent=2))
    print('Report size:',(OUT/'panel-report.html').stat().st_size)


if __name__ == '__main__':
    main()
