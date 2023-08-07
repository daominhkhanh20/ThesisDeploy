import axios from 'axios';

export const getAnswer = async (data) => {
    try{
        return await axios.post('http://149.36.0.24:30886/answer', data, {
            headers: {
                "Access-Control-Allow-Credentials": "true", 
                "Content-Type": "application/json"
            }
        });
    }catch(error){
        console.log(error);
    }
}