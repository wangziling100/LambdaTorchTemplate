const handler = require('../app').handler
test('handler', ()=>{
    const event = {
        message1: 'message1'
    }
    expect(handler(event)).toEqual({
        statusCode: 200,
        body: JSON.stringify({
            message: 'Hello World!'
        })
    })
})