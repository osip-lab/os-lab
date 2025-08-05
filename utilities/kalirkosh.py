import pyautogui

from utilities.general_gui_controller import *

BOX_LL = (3, 1072)  # get_cursor_position("lower-left corner of the tafnit window")
BOX_UR = (1916, 4)  # get_cursor_position("upper-right corner of the tafnit window")
SLEEP_TIME = 0.2
LONG_SLEEP_TIME = 2
# %% got to pritim tab:

button_position = detect_position('ivrit - main.png', relative_position=(0.5, 0.3),
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(button_position)
time.sleep(LONG_SLEEP_TIME)

button_position = detect_position('yazam.png', relative_position=(0.5, 0.3),
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(button_position)
time.sleep(LONG_SLEEP_TIME)

button_position = detect_position('ivrit - secondary.png',
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(button_position)
time.sleep(LONG_SLEEP_TIME)

button_position = detect_position('klita.png',
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(button_position)
pyautogui.moveTo(2, 2)
time.sleep(LONG_SLEEP_TIME)

button_position = detect_position('hazmana kaspit sherutim.png',
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(button_position)
time.sleep(8)

pritim_position = detect_position('pritim.png',
                                  crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(pritim_position)
time.sleep(LONG_SLEEP_TIME)

# %%
teur_position = detect_position('teur.png', relative_position=(-1, 0.5),
                                crop_ll=BOX_LL, crop_ur=BOX_UR)

kamut_position = detect_position('kamut.png', relative_position=(-1, 0.5),
                                 crop_ll=BOX_LL, crop_ur=BOX_UR)

mechir_bematbea_position = detect_position('mechir bematbea.png', relative_position=(-1, 0.5),
                                           crop_ll=BOX_LL, crop_ur=BOX_UR)

adken_shura_position = detect_position('adken shura.png',
                                       crop_ll=BOX_LL, crop_ur=BOX_UR)

# %%
category_1_position = detect_position('categories.png', relative_position=(-0.4, 0.85),
                                      crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(category_1_position)
time.sleep(SLEEP_TIME)

sherutim_position = detect_position('sherutim.png', relative_position=(0.9, 0.5),
                                    crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(sherutim_position)
time.sleep(SLEEP_TIME)

category_2_position = detect_position('categories.png', relative_position=(-0.4, 0.65),
                                      crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(category_2_position)
time.sleep(SLEEP_TIME)

eruim_kibud_achzaka_position = detect_position('eruim kibud achzaka.png',
                                               crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(eruim_kibud_achzaka_position)
time.sleep(SLEEP_TIME)

category_3_position = detect_position('categories.png', relative_position=(-0.4, 0.35),
                                      crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(category_3_position)
time.sleep(SLEEP_TIME)

kibud_kal_position = detect_position('kibud kal.png',
                                     crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(kibud_kal_position)
time.sleep(SLEEP_TIME)


# %%


def paste_row_to_fields(row):
    """Pastes values from a DataFrame row into designated screen fields."""
    # Paste each value to its corresponding field
    print(row)
    paste_value(row['Description'], teur_position)
    sleep(SLEEP_TIME)
    paste_value(row['Quantity'], kamut_position)
    sleep(SLEEP_TIME)
    paste_value(row['Price'], mechir_bematbea_position)
    sleep(SLEEP_TIME)
    pyautogui.click(category_1_position)
    sleep(SLEEP_TIME)
    pyautogui.click(sherutim_position)
    sleep(SLEEP_TIME)
    pyautogui.click(category_2_position)
    sleep(SLEEP_TIME)
    pyautogui.click(eruim_kibud_achzaka_position)
    sleep(SLEEP_TIME)
    pyautogui.click(category_3_position)
    sleep(SLEEP_TIME)
    pyautogui.click(kibud_kal_position)
    sleep(SLEEP_TIME)
    pyautogui.click(adken_shura_position)
    sleep(SLEEP_TIME)
    pyautogui.press('enter')
    sleep(SLEEP_TIME)


df = pd.read_csv(
    r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Michael Kalis files\GitProjects\Sticks\general-programs-controller\templates\halil order.csv")

for _, sample_row in df.iterrows():
    paste_row_to_fields(sample_row)
# %%
nispachim_position = detect_position('nispachim.png',
                                     crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(nispachim_position)
sleep(3)

sherutei_archive_position = detect_position('sherutei archive.png',
                                            crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(sherutei_archive_position)
sleep(LONG_SLEEP_TIME)
# %%
teur_mismach_position = detect_position('teur mismach.png',
                                        crop_ll=BOX_LL, crop_ur=BOX_UR, relative_position=(0.1, 0.5))
pyautogui.click(teur_mismach_position)
sleep(LONG_SLEEP_TIME)

pyautogui.write('quote')

haalaa_lasharat_position = detect_position('haalaa lasharat.png',
                                           crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(haalaa_lasharat_position)
sleep(LONG_SLEEP_TIME)
# %%
bechar_kovets_position = detect_position('bechar kovets.png',
                                         crop_ll=BOX_LL, crop_ur=BOX_UR)
pyautogui.click(bechar_kovets_position)
sleep(LONG_SLEEP_TIME)
# %%
ishur_upload_position = detect_position('ishur - upload.png',
                                        crop_ll=BOX_LL, crop_ur=BOX_UR, relative_position=(0.8, 0.5))
pyautogui.click(ishur_upload_position)
sleep(LONG_SLEEP_TIME)