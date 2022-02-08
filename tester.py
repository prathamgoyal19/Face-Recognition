import time
start_time=time.time()
import pickle
from final_embedding import extract_face

model=pickle.load(open("svm_model.sav","rb"))
in_encoder=pickle.load(open("in_encoder.pkl","rb"))
out_encoder=pickle.load(open("out_encoder.pkl","rb"))
emb=extract_face("salman.jpg")
emb=in_encoder.transform(emb)

pred=model.predict(emb)
print(out_encoder.inverse_transform(pred))
print(time.time()-start_time)