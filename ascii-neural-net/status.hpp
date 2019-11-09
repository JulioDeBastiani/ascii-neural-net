#pragma once

#include <string>

// TODO custom assert macro

namespace ann
{
    // ? err level
    class Status
    {
    public:
        enum error_codes
        {
            PLACEHOLDER = 0,
            INCOMPATIBLE_SIZES = 1,
            DESERIALIZATION_ERROR = 2
        };
    
        static Status OK()
        {
            return Status();
        }

        static Status ERROR(error_codes code, std::string message)
        {
            return Status(code, message);
        }

        bool ok() const
        {
            return _ok;
        }

        bool err() const
        {
            return !_ok;
        }

        int code() const
        {
            return _code;
        }

        const std::string& mesage() const
        {
            return _message;
        }

        std::string to_string() const
        {
            // TODO this could be better
            return _message;
        }

    private:
        bool _ok;
        int _code;
        std::string _message;

        Status():
            _ok(true)
        {
        }

        Status(error_codes code, std::string message):
            _ok(false),
            _code(code),
            _message(message)
        {
        }

        // ? file, line & stacktrace
        // * boost might be useful for this
    };
}