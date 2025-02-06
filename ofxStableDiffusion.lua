if type(window) ~= "userdata" then
window = ofWindow()
end

local a = ofelia
local canvas = ofCanvas(this)
local clock = ofClock(this, "setup")
local stableDiffusion = ofxStableDiffusion()
local settings = ofGLFWWindowSettings()
settings.visible = false
local modelName
local imgVec
local sampleMethod
local sampleMethodEnum
local textureTable = {}
local pixels = ofPixels()
local gui = ImGuiGui()
local boolArrayValue = ImGuiNew_BoolArray(1)
local intArrayValue = ImGuiNew_IntArray(1)
local charArray = ImGuiNew_CharPArray(8)
local intArray = ImGuiNew_IntArray(0)
local prompt = "a skater in the woods, van gogh"
local negativePrompt = ""
local generate = false
local send = ofSend("$0-goo")
local dataDir = canvas:getDir() .. "/data/"

function a.new()
ofWindow.addListener("setup", this)
ofWindow.addListener("update", this)
ofWindow.addListener("draw", this)
ofWindow.addListener("exit", this)
window:setPosition(200, 100)
window:setSize(632, 900)
if ofWindow.exists then
clock:delay(0)
else
window:createGLFW(settings)
end
end

function a.free()
window:destroy()
ofWindow.removeListener("setup", this)
ofWindow.removeListener("update", this)
ofWindow.removeListener("draw", this)
ofWindow.removeListener("exit", this)
end

function a.setup()
ofSetWindowTitle("ofxStableDiffusion")
ofBackground(150, 230, 255, 255)
for i = 1, 4 do
local texture = ofTexture()
texture:allocate(512, 512, GL_RGB)
textureTable[i] = texture
end
ImGuiBoolArray_setitem(boolArrayValue, 0, true)
ImGuiIntArray_setitem(intArrayValue, 0, 5)
local charTable = {"EULER_A", "EULER", "HEUN", "DPM2", "DPMPP2S_A", "DPMPP2M", "DPMPP2Mv2", "LCM"}
for i = 1, #charTable, 1 do
ImGuiCharPArray_setitem(charArray, i -1, charTable[i])
end
sampleMethod = ImGuiCharPArray_getitem(charArray, 0)
sampleMethodEnum = 0
print(ImGuiConfigFlags_ViewportsEnable)
gui:setup(ofxBaseTheme, true, ImGuiConfigFlags_ViewportsEnable)
modelName = "model/sd_turbo.safetensors"
modelPath = dataDir .. modelName
print(stableDiffusion:getSystemInfo())
stableDiffusion:newSdCtx(modelPath, "", "", "", "", "", "", "", "", "", "", true, false, false, 8, 1, 0, 0, false, false, false, false)
end

function a.update()
if stableDiffusion:isDiffused() then
imgVec = stableDiffusion:returnImages()
print("Width:", imgVec.width, "Height:", imgVec.height, "Channel:", imgVec.channel)
for i = 1, 4 do
textureTable[i]:loadData(stableDiffusion:getImageAt(i - 1).data, 512, 512, GL_RGB)
end
stableDiffusion:setDiffused(false)
end
if generate then
stableDiffusion:txt2img(prompt, negativePrompt, 0, 1, 0, 512, 512, sampleMethodEnum, 5, -1, 4, sd_image_t, 1, 1, false, "", intArray, 0, 0, 0, 0)
generate = false
end
end

function a.draw()
gui:beginGui()
--ImGuiShowDemoWindow_0()
ImGuiStyleColorsDark()
ImGuiPushStyleVar(ImGuiStyleVar_WindowMinSize, ImGuiImVec2(532, 120))
ImGuiPushStyleVar(ImGuiStyleVar_WindowPadding, ImGuiImVec2(10, 0))
ImGuiPushStyleVar(ImGuiStyleVar_IndentSpacing, 10)
ImGuiPushStyleVar(ImGuiStyleVar_ItemSpacing, ImGuiImVec2(0, 0))
ImGuiPushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImGuiImVec2(5, 0))
ImGuiSetNextWindowPos(ImGuiImVec2(50, 50), ImGuiCond_Once)
ImGuiBegin_3("ofxStableDiffusion", boolArrayValue, ImGuiWindowFlags_NoResize)
ImGuiDummy(ImGuiImVec2(0, 10))
ImGuiImage(textureTable[1]:getTextureData().textureID, ImGuiImVec2(256, 256))
ImGuiSameLine()
ImGuiImage(textureTable[2]:getTextureData().textureID, ImGuiImVec2(256, 256))
ImGuiImage(textureTable[3]:getTextureData().textureID, ImGuiImVec2(256, 256))
ImGuiSameLine()
ImGuiImage(textureTable[4]:getTextureData().textureID, ImGuiImVec2(256, 256))
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiButton("Save")) then
textureTable[1]:readToPixels(pixels)
ofSaveImage(pixels, ofGetTimestampString(dataDir .. "output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"))
end
ImGuiDummy(ImGuiImVec2(0, 10))
ImGuiPushItemWidth(420)
ImGuiText("The checkbox below is checked.")
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiSliderInt("Test", intArrayValue, 1, 16)) then
send:sendFloat(ImGuiIntArray_getitem(intArrayValue, 0))
end
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiCheckbox("Checkbox", boolArrayValue)) then
print("Toggle:", ImGuiBoolArray_getitem(boolArrayValue, 0))
end
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiRadioButton("Checked", true)) == true then
print("Button pressed!")
end
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiBeginCombo("Sample Method", sampleMethod, ImGuiComboFlags_NoArrowButton)) then
for i = 0, 7, 1 do
local isSelected = ImGuiNew_BoolArray(1)
ImGuiBoolArray_setitem(isSelected, 0, sampleMethod == ImGuiCharPArray_getitem(charArray, i))
if (ImGuiSelectable(ImGuiCharPArray_getitem(charArray, i), isSelected)) then
sampleMethod = ImGuiCharPArray_getitem(charArray, i)
sampleMethodEnum = i
end
end
ImGuiEndCombo()
end
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiButton("Load Model")) then
local result = ofSystemLoadDialog("Load Model", false, "")
if (result.bSuccess) then
local modelPath = result:getPath()
modelName = result:getName()
local file = ofFile(result:getPath())
if(file:getExtension() == "safetensors") then
stableDiffusion:newSdCtx(modelPath, "", "", "", "", "", "", "", "", "", "", true, false, false, 8, 1, 0, 0, false, false, false, false)
end
end
end
ImGuiSameLine(0, 5)
ImGuiText(modelName)
ImGuiDummy(ImGuiImVec2(0, 10))
if (ImGuiButton("Generate")) then
generate = true
print("Prompt:", prompt)
end
ImGuiDummy(ImGuiImVec2(0, 10))
ImGuiInputText( "Prompt", prompt, 60)
ImGuiDummy(ImGuiImVec2(0, 10))
ImGuiInputText( "nPrompt", negativePrompt, 60)
ImGuiDummy(ImGuiImVec2(0, 20))
ImGuiEnd()
ImGuiPopStyleVar()
ImGuiPopStyleVar()
ImGuiPopStyleVar()
ImGuiPopStyleVar()
ImGuiPopStyleVar()
gui:endGui()
end

function a.exit()
gui:exit();
stableDiffusion:freeSdCtx()
ImGuiDelete_BoolArray(boolArrayValue)
ImGuiDelete_IntArray(intArrayValue)
ImGuiDelete_CharPArray(charArray)
end