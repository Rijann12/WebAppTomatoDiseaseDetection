from flask import Flask,request,jsonify
import google.generativeai as genai

app = Flask(__name__)
#Insert Gemini API key 
genai.configure(api_key="AIzaSyCNOj54zXIYg2N4tQni5rNfiHL5K-i5I6o")

model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/precaution', methods=['POST'])
def get_precaution():
    data= request.json
    print(" Received request:", data)

    disease = data.get("disease","")
    
    if not disease:
        return jsonify({"error": "Disease name is required"}),400 
    
     # Special handling
    if disease == "Tomato___healthy" or "healthy" in disease.lower():
        return jsonify({"Precaution": "The provided leaf is healthy. No remedies are required."})
    
    if disease.lower() in ["not leaf", "not known", "unknown"]:
        return jsonify({"Precaution": "The provided image is either not a leaf or represents an unknown disease. No precaution can be provided."})
    
    prompt = (
    f"Act like a veteran agricultural doctor. "
    f"Give very simple, practical, and useful precaution and treatment tips for {disease}, "
    f"especially tailored for farmers. Avoid technical language. Be direct, clear, and helpful. "
    )

    try:
        response = model.generate_content(prompt)
        print(" Gemini response:", response.text)
        return jsonify({"Precaution": response.text})
    
    except Exception as e:
        print(" Gemini suggestion:", e)
        return jsonify({"error":str(e)}), 500
if __name__=='__main__' :
    app.run(host='0.0.0.0', port=5001, debug=True)
  
    