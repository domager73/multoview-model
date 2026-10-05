# Multiview model playground

Интерактивный сайт: генерирует multiview-смесь, восстанавливает её моментным методом,
EM и гибридом, показывает графики и метрики. Считается в браузере через Pyodide
(настоящие numpy/scipy) — серверу Python не нужен.

## Запуск локально
    python3 -m http.server 8123
    # открыть http://localhost:8123

(нужен интернет — Pyodide подгружается с CDN; сам расчёт локальный)

## Публикация на GitHub Pages
    cd site
    git init && git add -A && git commit -m "multiview playground"
    git branch -M main
    git remote add origin git@github.com:<username>/multiview-playground.git
    git push -u origin main
Затем Settings -> Pages -> Source: Deploy from branch -> main / root.
Сайт: https://<username>.github.io/multiview-playground/

## Что регулируется
распределение (gauss/beta/heavy/uniform/laplace), размерность d, число компонент r,
число шапок L, размер базиса K, объём выборки n, сглаживание gamma, контаминация eps, seed.
