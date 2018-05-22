# coding: utf-8
import sys

class ReactTemplate:

    page = """
import React, { Component } from 'react';
import ReactDOM from 'react-dom';

//import using commonJS Module *Require Plugins
//import { Button } from 'react-weui'

//import Using ES6 syntax
import WeUI from 'react-weui';

//import styles
import 'weui';
import 'react-weui/lib/react-weui.min.css';

const { ButtonArea,
    Button,
    CellsTitle,
    CellsTips,
    Cell,
    CellHeader,
    CellBody,
    CellFooter,
    Form,
    FormCell,
    Icon,
    Input,
    Label,
    TextArea,
    Switch,
    Radio,
    Checkbox,
    Select,
    VCode,
    Agreement,
    Toptips,
    Page,
    Slider,

} = WeUI

class App extends Component {
    constructor(props) {
        super(props)
    }
    render() {
        // var page = random_page()
        // console.log(page[1])
        return (
        <div>
            %s
        </div>
        )
    }
}
export default App;
"""

    cell_button = """
<ButtonArea direction="horizontal">%s</ButtonArea>
"""
    button = """
<Button>Button</Button>"""

    cell = """
<Form>%s
</Form>"""

    textarea = """
<FormCell>
    <CellBody>
        <TextArea rows="3" placeholder="textarea content" />
    </CellBody>
</FormCell>"""

    input = """
<FormCell>
    <CellHeader>        
        <Label>Label</Label>
    </CellHeader>
    <CellBody>
        <Input placeholder="input content" />
    </CellBody>
</FormCell>"""	

    input_select = """
<FormCell select selectPos="before">
    <CellHeader>
        <Select>
            <option value="1">option</option>
        </Select>
    </CellHeader>
    <CellBody>
        <Input type="tel" placeholder="input content"/>
    </CellBody>
</FormCell>"""

    input_button = """
<FormCell>
    <CellHeader>        
        <Label>Lable</Label>
    </CellHeader>
    <CellBody>
        <Input placeholder="input content" />
    </CellBody>
    <CellFooter>
        <Button type="vcode">Button"</Button>
    </CellFooter>
</FormCell>"""

    cell_radio = """
<Form radio>%s</Form>"""

    radio = """
<FormCell radio>
    <CellBody>Radio</CellBody>
    <CellFooter>
        <Radio name="radio1" value="1" defaultChecked/>
    </CellFooter>
</FormCell>
<FormCell radio>
    <CellBody>Radio</CellBody>
    <CellFooter>
        <Radio name="radio1" value="2"/>
    </CellFooter>
</FormCell>"""

    cell_checkbox = """
<Form checkbox>%s</Form>"""

    checkbox = """
<FormCell checkbox>
    <CellHeader>
        <Checkbox name="checkbox1" value="1" defaultChecked/>
    </CellHeader>
    <CellBody>checkbox</CellBody>
</FormCell>
<FormCell checkbox>
    <CellHeader>
        <Checkbox name="checkbox2" value="2" />
    </CellHeader>
    <CellBody>checkbox</CellBody>
</FormCell>"""

    switch = """
<FormCell switch>
    <CellBody>Switch</CellBody>
    <CellFooter>
        <Switch/>
    </CellFooter>
</FormCell>"""

    cell_title = """
<CellsTitle>Title</CellsTitle>"""
    

    @classmethod
    def temp(self, name, tab):
        if ReactTemplate.__dict__.has_key(name):
            return ReactTemplate.__dict__[name].replace("\n", "\n"+tab)
        else:
            return ""
    

START_TAG = "start"
END_TAG = "end"

class Decoder:
    start_tag = START_TAG
    end_tag = END_TAG
    block_tags = ["cell"]
    form_tags = ["input", "input_select", "cell_title", "switch", "checkbox", "button", "option", "textarea", "input_button"]
    block_form_tags = ["cell", "cell_title"]

    def decode(self, tags):
        page = ptr = Component("page", True)
        for tag in tags:
            if tag == START_TAG:
                continue
            if tag == END_TAG:
                break
            if tag in self.block_tags:
                is_block = True
            else:
                is_block = False
            
            comp = Component(tag, is_block)
            if ptr.tag_name == "page":
                ptr.appendChild(comp)
                if tag in self.block_tags:
                    ptr = comp
                continue

            if is_block and tag == ptr.tag_name and len(ptr.children) == 0:
                continue

            if tag in self.block_form_tags:
                ptr.parent.appendChild(comp)
                ptr = comp
                continue

            ptr.appendChild(comp)

        return Page(page)

class Component:
    def __init__(self, tag_name, is_block):
        self.tag_name = tag_name
        self.c_type = is_block
        self.parent = None
        self.children = []

    def appendChild(self, comp):
        self.children.append(comp)
        comp.parent = self

    def is_root(self):
        return self.parent == None 
    def __str__(self):
        return self.tag_name

class Page:
    def __init__(self, comps):
        self.comps = comps
    
    def __str__(self):
        ptr = self.comps
        r = ""
        tab = ""
        def each(ptr, tab, r):
            if len(ptr.children) > 0:
                r += "%s%s {\n" % (tab, ptr.tag_name)
                for comp in ptr.children:
                    r = each(comp, tab+"  ", r)
                r += "%s}\n" % tab
            else:
                r += "%s%s\n" % (tab, ptr.tag_name)
            return r

        r = each(ptr, tab, r)
        return r
    
    def react(self):
        ptr = self.comps
        r = ""
        tab = ""
        def each(ptr, tab, r):
            if len(ptr.children) > 0:
                rs = "" 
                for comp in ptr.children:
                    rs = each(comp, tab+"    ", rs)
                if ptr.children[0].tag_name == "button":
                    temp = ReactTemplate.temp("cell_button", tab)
                else:
                    temp = ReactTemplate.temp(ptr.tag_name, tab)
                r += temp % rs 
            else:
                r += ReactTemplate.temp(ptr.tag_name, tab)
            return r

        r = each(ptr, tab, r)
        return r
         
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        tags = "start,cell_title,cell,input,input_select,input,cell,switch,cell,input_button,input_button,input_button,input_select,end"
    else:
        tags = sys.argv[1]
    
    decoder = Decoder()
    page = decoder.decode(tags.split(","))
    print page
    page = page.react()

    with open("ReactApp/show-app/src/App.js", "w") as f:
        f.write(page)
    print "write react js file finish"

