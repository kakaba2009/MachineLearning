import re


def fun(s):
    # return True if s is a valid email, else return False
    f = "^[a-zA-Z][\w-]*@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$"
    if not re.match(f, s):
        return False
    username, after = re.split(r'[@]', s)
    websitename, extension = re.split(r'[.]', after)
    if(len(extension) > 3):
        return False
    return True


def filter_mail(emails):
    return list(filter(fun, emails))

if __name__ == '__main__':
    n = int(input())
    emails = []
    for _ in range(n):
        emails.append(input())

filtered_emails = filter_mail(emails)
filtered_emails.sort()
print(filtered_emails)
