const k1 = process.env.k1
exports.handler= (event, context) =>{
    console.log('event: ', event)
    console.log('env variable: ', k1)
    const response = {
        'statusCode': 200,
        'body': JSON.stringify({
            message: 'Hello World!'
        })
    }
    return response
}