import argparse
import copy
import json
import re
import sys
import types
import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "Gem5 to McPAT parser")

    parser.add_argument(
        '--config', '-c', type = str, required = True,
        metavar = 'PATH',
        help = "Input config.json from Gem5 output.")
    parser.add_argument(
        '--stats', '-s', type = str, required = True,
        metavar = 'PATH',
        help = "Input stats.txt from Gem5 output.")
    parser.add_argument(
        '--template', '-t', type = str, required = True,
        metavar = 'PATH',
        help = "Template XML file")
#    parser.add_argument(
#        '--boom', '-b', type = str, required = True,
#        metavar = 'PATH',
#        help = "Input boom.xml from BOOM design.")
    parser.add_argument(
        '--output', '-o', type = argparse.FileType('w'), default = "mcpat-in.xml",
        metavar = 'PATH',
        help = "Output file for McPAT input in XML format (default: mcpat-in.xml)")
    
    return parser


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class PIParser(ET.XMLTreeBuilder):
   def __init__(self):
       ET.XMLTreeBuilder.__init__(self)
       # assumes ElementTree 1.2.X
       self._parser.CommentHandler = self.handle_comment
       self._parser.ProcessingInstructionHandler = self.handle_pi
       self._target.start("document", {})

   def close(self):
       self._target.end("document")
       return ET.XMLTreeBuilder.close(self)

   def handle_comment(self, data):
       self._target.start(ET.Comment, {})
       self._target.data(data)
       self._target.end(ET.Comment)

   def handle_pi(self, target, data):
       self._target.start(ET.PI, {})
       self._target.data(target + " " + data)
       self._target.end(ET.PI)


def parse_xml(source):
    return ET.parse(source, PIParser())


def read_stats(stats_file):
    global stats
    stats = {}
    with open(stats_file, 'r') as f:
        ignores = re.compile(r'^---|^$')
        stat_line = re.compile(r'([a-zA-Z0-9_\.:-]+)\s+([-+]?[0-9]+\.[0-9]+|[-+]?[0-9]+|nan|inf)')
        count = 0
        for line in f:
            # ignore empty lines and lines starting with "---"
            if not ignores.match(line):
                count += 1
                stat_kind = stat_line.match(line).group(1)
                stat_value = stat_line.match(line).group(2)
                if stat_value == 'nan':
                    print("[WARN]: %s is nan. Setting it to 0." % stat_kind)
                    stat_value = '0'
                stats[stat_kind] = stat_value


def read_config(config_file):
    global config
    with open(config_file, 'r') as f:
        config = json.load(f)


def read_mcpat_template(template):
    global mcpat_template 
    mcpat_template = parse_xml(template)


def read_boom(boom_paramsFile):
    global boom_params
    boom_params = parse_xml(boom_paramsFile)


def generate_template(output_file):
    num_cores = len(config["system"]["cpu"])
    private_l2 = config["system"]["cpu"][0].has_key('l2cache')
    shared_l2 = config["system"].has_key('l2')
    if private_l2:
        num_l2 = num_cores
    elif shared_l2:
        num_l2 = 1
    else:
        num_l2 = 0
    elem_counter = 0
    root = mcpat_template.getroot()
    for child in root[0][0]:
        # to add elements in correct sequence
        elem_counter += 1 

        if child.attrib.get("name") == "number_of_cores":
            child.attrib['value'] = str(num_cores)
        if child.attrib.get("name") == "number_of_L2s":
            child.attrib['value'] = str(num_l2)
        if child.attrib.get("name") == "Private_L2":
            if shared_l2:
                Private_L2 = str(0)
            else:
                Private_L2 = str(1)
            child.attrib['value'] = Private_L2
        temp = child.attrib.get('value')

        # to consider all the cpus in total cycle calculation
        if num_cores > 1 and isinstance(temp, basestring) and "cpu." in temp and temp.split('.')[0] == "stats":
            value = "(" + temp.replace("cpu.", "cpu0.") + ")"
            for i in range(1, num_cores):
                value = value + " + (" + temp.replace("cpu.", "cpu"+str(i)+".") +")"
            child.attrib['value'] = value

        # remove a core template element and replace it with number of cores template elements
        if child.attrib.get("name") == "core":
            coreElem = copy.deepcopy(child)
            coreElemCopy = copy.deepcopy(coreElem)
            for coreCounter in range(num_cores):
                coreElem.attrib["name"] = "core" + str(coreCounter)
                coreElem.attrib["id"] = "system.core" + str(coreCounter)
                for coreChild in coreElem:
                    childId = coreChild.attrib.get("id")
                    childValue = coreChild.attrib.get("value")
                    childName = coreChild.attrib.get("name")
                    if isinstance(childName, basestring) and childName == "x86":
                        if config["system"]["cpu"][coreCounter]["isa"][0]["type"] == "X86ISA":
                            childValue = "1"
                        else:
                            childValue = "0"
                    if isinstance(childId, basestring) and "core" in childId:
                        childId = childId.replace("core", "core" + str(coreCounter))
                    if num_cores > 1 and isinstance(childValue, basestring) and "cpu." in childValue and "stats" in childValue.split('.')[0]:
                        childValue = childValue.replace("cpu." , "cpu" + str(coreCounter)+ ".")
                    if isinstance(childValue, basestring) and "cpu." in childValue and "config" in childValue.split('.')[0]:
                        childValue = childValue.replace("cpu." , "cpu." + str(coreCounter)+ ".")
                    if len(list(coreChild)) != 0:
                        for level2Child in coreChild:
                            level2ChildValue = level2Child.attrib.get("value")
                            if num_cores > 1 and isinstance(level2ChildValue, basestring) and "cpu." in level2ChildValue and "stats" in level2ChildValue.split('.')[0]:
                                level2ChildValue = level2ChildValue.replace("cpu." , "cpu" + str(coreCounter)+ ".")
                            if isinstance(level2ChildValue, basestring) and "cpu." in level2ChildValue and "config" in level2ChildValue.split('.')[0]:
                                level2ChildValue = level2ChildValue.replace("cpu." , "cpu." + str(coreCounter)+ ".")
                            level2Child.attrib["value"] = level2ChildValue
                    if isinstance(childId, basestring):
                        coreChild.attrib["id"] = childId
                    if isinstance(childValue, basestring):
                        coreChild.attrib["value"] = childValue
                root[0][0].insert(elem_counter, coreElem)
                coreElem = copy.deepcopy(coreElemCopy)
                elem_counter += 1
            root[0][0].remove(child)
            elem_counter -= 1

        # # remove a L2 template element and replace it with the private L2 template elements
        # if child.attrib.get("name") == "L2.shared":
        #     print(child)
        #     if shared_l2:
        #         child.attrib["name"] = "L20"
        #         child.attrib["id"] = "system.L20"
        #     else:
        #         root[0][0].remove(child)

        # remove a L2 template element and replace it with number of L2 template elements
        if child.attrib.get("name") == "L2":
            if private_l2:
                print("private_l2")
                l2Elem = copy.deepcopy(child)
                l2ElemCopy = copy.deepcopy(l2Elem)
                for l2Counter in range(num_l2):
                    l2Elem.attrib["name"] = "L2" + str(l2Counter)
                    l2Elem.attrib["id"] = "system.L2" + str(l2Counter)
                    for l2Child in l2Elem:
                        childValue = l2Child.attrib.get("value")
                        if isinstance(childValue, basestring) and "cpu." in childValue and "stats" in childValue.split('.')[0]:
                            childValue = childValue.replace("cpu." , "cpu" + str(l2Counter)+ ".")
                        if isinstance(childValue, basestring) and "cpu." in childValue and "config" in childValue.split('.')[0]:
                            childValue = childValue.replace("cpu." , "cpu." + str(l2Counter)+ ".")
                        if isinstance(childValue, basestring):
                            l2Child.attrib["value"] = childValue
                    root[0][0].insert(elem_counter, l2Elem)
                    l2Elem = copy.deepcopy(l2ElemCopy)
                    elem_counter += 1
                root[0][0].remove(child)
            else:
                print("not private_l2")
                child.attrib["name"] = "L20"
                child.attrib["id"] = "system.L20"
                for l2Child in child:
                    childValue = l2Child.attrib.get("value")
                    if isinstance(childValue, basestring) and "cpu.l2cache." in childValue:
                        childValue = childValue.replace("cpu.l2cache." , "l2.")

    prettify(root)


def get_config_value(confStr):
    split_conf = re.split('\.', confStr)
    curr_conf = config
    curr_hierarchy = ""
    for x in split_conf:
        curr_hierarchy += x
        if x.isdigit():
            curr_conf = curr_conf[int(x)] 
        elif x in curr_conf:
            curr_conf = curr_conf[x]
    if(curr_conf == None):
        return 0
        curr_hierarchy += "."
    return curr_conf


def write_mcpat_xml(output_path):
    root_elem = mcpat_template.getroot()
#    boomroot_elem = boom_params.getroot()
    pattern = re.compile(r'config\.([][a-zA-Z0-9_:\.]+)')
    #replace params with values from the GEM5 config file 
    for param in root_elem.iter('param'):
        name = param.attrib['name']
        value = param.attrib['value']
        if 'config' in value:
            allConfs = pattern.findall(value)
            for conf in allConfs:
                confValue = get_config_value(conf)
                if type(confValue) == dict or type(confValue) == list :
                    confValue = 0
                    print("[WARN]: %s does not exist in gem5 config." % conf)
                value = re.sub("config."+ conf, str(confValue), value)
            if "," in value:
                exprs = re.split(',', value)
                for i in range(len(exprs)):
                    exprs[i] = str(eval(exprs[i]))
                param.attrib['value'] = ','.join(exprs)
            else:
                param.attrib['value'] = str(eval(str(value)))
        elif 'boom' in value:
            for boom_param in boomroot_elem.iter('param'):
                boom_name = boom_param.attrib['name']
                boom_value = boom_param.attrib['value']
                if (name == boom_name):
                    value = boom_value
                    param.attrib['value'] = str(eval(str(value)))
                    break
            if 'boom' in value:
                print("[WARN]:%s does not exist in boom config." % boom)


    #replace stats with values from the GEM5 stats file 
    statRe = re.compile(r'stats\.([a-zA-Z0-9_:\.]+)')
    for stat in root_elem.iter('stat'):
        name = stat.attrib['name']
        value = stat.attrib['value']
        if 'stats' in value:
            allStats = statRe.findall(value)
            expr = value
            for i in range(len(allStats)):
                if allStats[i] in stats:
                    expr = re.sub('stats.%s' % allStats[i], stats[allStats[i]], expr)
                else:
                    expr = re.sub('stats.%s' % allStats[i], str(0), expr)
                    print("[WARN]: %s does not exist in stats." % allStats[i])

            if 'config' not in expr and 'stats' not in expr:
                try:
                    stat.attrib["value"] = str(eval(expr))
                except ZeroDivisionError as e:
                    print("[ERROR]: %s" % e)

    #Write out the xml file
    mcpat_template.write(output_path)            

def main():
    read_stats(args.stats)
    read_config(args.config)
#    read_boom(args.boom)
    read_mcpat_template(args.template)
    generate_template(args.output)
    write_mcpat_xml(args.output)

if __name__ == '__main__':
    args = create_parser().parse_args()
    main()
