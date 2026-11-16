# Upgrading the website

This implementation will add new features but should not ruin the webpage

## Header
Inser links as buttons. They should not be the first thing to catch the eye. Simple, in the theme of website blue buttons. Left is the icon, then the text. First button begins right above the upload image box

Rules (this is the one that stands out and not a link). Then GitHub, Colab, YouTube, Writeup
Their order and links are in the website_parameters file

## Model names and descriptions
Given from the website_parametes file

## Model 5 slight RGB glowing
Relative size or false from website_parameters

## Benchmarking
View benchmarks opens a beautiful table.
Look at the reference image user uploads if any. This is the benchmarking table.

**HERE:** The green fields get colored, other colors only circle in the color of benchmark. THE BEST MODEL (V5) Getes circled across the column in blue and filled title card while others dont have filled title card

Buttons to close the table: Escape works, x button, OK button, and view extended benchmarking (which implements the boring benchmarks_full table). There are buttons Simple benchmarking, OK, and x (esc works)

## The rules
Rules button on the header - not a link.
This will need to implement the rules of the model, explained simply. Think of proper colors here. See user upload if any.

This table is inBULGARIAN, but everything sould be in english
### Термини, използвани в таблицата:
* **Валидност (`Scope`):** Вероятността (според Gemini), че снимката е рентген на гръден кош. Минималният праг е **`scope_min` (50%)**.
* **Качество (`Quality`):** Техническата оценка на Gemini — `High`, `Noisy` или `Unacceptable`. *(В текущата конфигурация `quality_blocks = False`, затова лошото качество само сваля нивото от **A** на **B**, без да спира анализа.)*
* **Праг на ResNet (`t_res`):** Калибрираната увереност, нужна на ResNet-18, за да се смята отговорът му за сигурен (изчислена на валидация за ≤10% грешка).
* **Праг на Gemini (`t_gem`):** Минималната увереност на Gemini за диагнозата (по подразбиране **80%**).
* **COVID Override (`t_covid`):** Специален висок праг на ResNet-18 за `COVID-19` (изчислен за ≥90% прецизност; по подразбиране **85%**).

---

### Tier-Table (Таблица на решенията)

Логиката се проверява **строго отгоре надолу** — първото изпълнено условие определя крайния **Tier**:

| Tier | Условие (Какво се гледа) | Съгласие (`agree`) | Риск (`risk`) | Какво вижда потребителят в уебсайта |
| :--- | :--- | :--- | :--- | :--- |
| **U** | **Gemini не е наличен** (грешка, timeout или липсва API ключ). | `N/A` | `LOW` / `HIGH` *(ако ResNet ≥ `t_res`)*<br>`REVIEW` *(ако ResNet < `t_res`)* | **Временен резултат само от 1 модел.** Излиза банер `"Scope not verified"`, защото не е проверено дали снимката изобщо е рентген. |
| **X** | **`Scope` < 50%** (не прилича на рентген на гръден кош) *(или `Unacceptable` с ≥70% увереност, ако `quality_blocks = True`)*. | `N/A` | `NONE` | **Спира анализа (`NO READING`).** Системата отказва диагноза и моли за нова снимка. |
| **A** | Двата модела са на **едно мнение** **И** ResNet ≥ `t_res` **И** Gemini ≥ `t_gem` **И** качеството е `High`. | `AGREE` | `LOW` *(NORMAL)*<br>`HIGH` *(болест)* | **Най-висока сигурност.** Пълно съгласие между двата модела без никакви забележки. |
| **B** | Двата модела са на **едно мнение**, но има **поне 1 забележка**: ResNet < `t_res`, ИЛИ Gemini < `t_gem`, ИЛИ качеството не е `High`. | `AGREE` | `LOW` *(NORMAL)*<br>`HIGH` *(болест)* | **Съгласие с предупреждение.** Резултатът важи, но излиза текст `"Caution:"` с точната причина (напр. шумна снимка или ниска увереност). |
| **C** | Моделите **се разминават**, но `covid_override = True`, ResNet казва `COVID-19` **И** увереността му е ≥ `t_covid`. | `DISAGREE` | `HIGH` | **Специален приоритет за COVID-19.** Доверието отива при ResNet-18, но се изисква потвърждение от лекар. |
| **D** | Моделите **се разминават** и не влизат в Tier C. | `DISAGREE` | `REVIEW` | **Задължителен човешки преглед.**<br>• `URGENT REVIEW`: и двата подозират болест, но различна (напр. Pneumonia vs COVID-19).<br>• `REVIEW`: единият казва `NORMAL`, а другият подозира болест. |

So translate into english and implement


## NO SLOP IN THE SITE. KEEP THE SIMPLICITY, INTEGRITY.

