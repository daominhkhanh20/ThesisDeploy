import axios from 'axios';

export const getAnswer = async (data) => {
    try{
        return await axios.post('http://0.0.0.0:5000/answer', data, {
            headers: {
                "Access-Control-Allow-Credentials": "true", 
                "Content-Type": "application/json"
            }
        });
    }catch(error){
        console.log(error);
    }
}