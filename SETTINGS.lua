require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

DATA_DIR="./data"

CLASSES = {}
LABEL2ID = {1,2,3,4,5,6,7,8,9,10}
ID2LABEL = {}
for k, v in pairs(LABEL2ID) do
   ID2LABEL[v] = k
   CLASSES[v] = k
end

return true
