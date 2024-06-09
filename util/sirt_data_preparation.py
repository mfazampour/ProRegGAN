import os
import subprocess


def modify_mdh_files(patients):
    iws_files = []
    for patient in patients:
        iws_file = patient['mr_path'].replace('Diagnostic/', 'workspace.iws')
        if not os.path.exists(iws_file):
            print(f'{iws_file} does not exist')
            continue
        with open(iws_file) as f:
            lines = f.readlines()
            edited_lines = []
            for line in lines:
                if '>CT<' in line:
                    edited_lines.append(line)
                    edited_lines.append('\t\t\t<param name="uid">data0</param>')
                elif 'Liver Label Map' in line:
                    edited_lines.append(line)
                    edited_lines.append('\t\t\t<param name="uid">data4</param>')
                elif '</propertyfile>' in line:
                    algo = '\t<property name="Algorithms"> \n' \
                           '\t\t<property name="Image Resampling"> \n' \
                           '\t\t\t<param name="createNewImage">1</param> \n' \
                           '\t\t\t<param name="forceCPU">0</param> \n' \
                           '\t\t\t<param name="preserveExtent">0</param> \n' \
                           '\t\t\t<param name="keepZeroValues">0</param> \n' \
                           '\t\t\t<param name="verbose">0</param> \n' \
                           '\t\t\t<param name="reductionMode">1</param> \n' \
                           '\t\t\t<param name="interpolationMode">1</param> \n' \
                           '\t\t\t<param name="execute">1</param> \n' \
                           '\t\t\t<param name="inputUids">"data0" "data4" </param> \n' \
                           '\t\t\t<param name="outputUids">"data9" </param> \n ' \
                           '\t\t</property> \n ' \
                           '\t\t<property name="MetaImage"> \n ' \
                           '\t\t\t<param name="location">CT_resampled.mhd</param> \n ' \
                           '\t\t\t<param name="multiFile">0</param> \n' \
                           '\t\t\t<param name="unsigned">0</param> \n' \
                           '\t\t\t<param name="compress">0</param> \n' \
                           '\t\t\t<param name="ignoreHalfPixelOffset">0</param> \n' \
                           '\t\t\t<param name="execute">1</param> \n' \
                           '\t\t\t<param name="inputUids">"data9" </param> \n' \
                           '\t\t\t<param name="outputUids"></param> \n' \
                           '\t\t</property> \n' \
                           '\t</property> '
                    algo = [l + '\n' for l in algo.split('\n')]
                    edited_lines += algo
                    edited_lines.append(line)
                else:
                    edited_lines.append(line)
        iws_new = iws_file.replace('workspace.iws', 'workspace_resample.iws')
        with open(iws_new, mode='w') as f:
            for line in edited_lines:
                f.write(line)


        iws_files.append(iws_new)
    return iws_files


def run_iws_file(iws_files):
    for path in iws_files:
        subprocess.run(["ImFusionConsole", f'{path}'])
